package kademlia

import (
    "context"
    "crypto/sha1"
    "fmt"
    "strings"
    "sync"
    "time"

    "github.com/libp2p/go-libp2p/core/host"
    "github.com/libp2p/go-libp2p/core/peer"
)

type Node struct {
    host               host.Host
    nodeIDManager      *NodeIDManager
    routing            *RoutingTable
    protocol           *ProtocolHandler
    maintenanceManager *MaintenanceManager
    ctx                context.Context
    cancel             context.CancelFunc
    isRunning          bool
    mutex              sync.RWMutex
}

func NewNode(host host.Host, dataDir string) (*Node, error) {
    // Create persistent node ID manager
    nodeIDManager, err := NewNodeIDManager(dataDir, host.ID())
    if err != nil {
        return nil, fmt.Errorf("failed to create node ID manager: %w", err)
    }

    ctx, cancel := context.WithCancel(context.Background())
    
    nodeID := nodeIDManager.GetNodeID()
    routing := NewRoutingTable(nodeID)
    protocol, err := NewProtocolHandler(host, routing, nodeID, dataDir)
    if err != nil {
        cancel()
        return nil, err
    }

    node := &Node{
        host:          host,
        nodeIDManager: nodeIDManager,
        routing:       routing,
        protocol:      protocol,
        ctx:           ctx,
        cancel:        cancel,
    }

    // Initialize maintenance manager
    node.maintenanceManager = NewMaintenanceManager(node)

    return node, nil
}

func (n *Node) Start() error {
    n.mutex.Lock()
    defer n.mutex.Unlock()
    
    if n.isRunning {
        return fmt.Errorf("node already running")
    }
    
    n.isRunning = true
    
    // Load persisted contacts
    if err := n.loadPersistedContacts(); err != nil {
        fmt.Printf("Warning: Failed to load contacts: %v\n", err)
    }
    
    // Start maintenance
    n.maintenanceManager.Start()
    
    fmt.Printf("GORM-based Kademlia node started\n")
    return nil
}

func (n *Node) Stop() {
    n.mutex.Lock()
    defer n.mutex.Unlock()
    
    if !n.isRunning {
        return
    }
    
    n.isRunning = false
    
    // Stop maintenance
    n.maintenanceManager.Stop()
    
    // Save contacts before shutdown
    n.saveAllContacts()
    
    // Close database
    n.protocol.dataStore.Close()
    
    n.cancel()
    fmt.Println("Kademlia node stopped")
}

func (n *Node) Bootstrap(bootstrapPeers []peer.ID) error {
    if len(bootstrapPeers) == 0 {
        fmt.Println("No bootstrap peers provided")
        return nil
    }

    fmt.Printf("Bootstrapping with %d peers...\n", len(bootstrapPeers))
    
    connected := 0
    for _, peerID := range bootstrapPeers {
        fmt.Printf("Connecting to bootstrap peer: %s\n", peerID)
        
        // Connect with timeout
        connectCtx, connectCancel := context.WithTimeout(n.ctx, 15*time.Second)
        err := n.host.Connect(connectCtx, peer.AddrInfo{ID: peerID})
        connectCancel()
        
        if err != nil {
            fmt.Printf("Failed to connect: %v\n", err)
            continue
        }
        
        // Extract real addresses
        var addrs []string
        if conns := n.host.Network().ConnsToPeer(peerID); len(conns) > 0 {
            remoteAddr := conns[0].RemoteMultiaddr().String()
            // Remove /p2p/peerID part
            if idx := strings.Index(remoteAddr, "/p2p/"); idx != -1 {
                addrs = append(addrs, remoteAddr[:idx])
            }
        }
        
        // Force add to routing table
        success := n.routing.AddContact(peerID, addrs)
        fmt.Printf("Added to routing table: %v, addrs: %v\n", success, addrs)
        
        // Try ping with retries
        for attempt := 0; attempt < 3; attempt++ {
            pingCtx, pingCancel := context.WithTimeout(n.ctx, 5*time.Second)
            err = n.Ping(pingCtx, peerID)
            pingCancel()
            
            if err == nil {
                fmt.Printf("Ping successful on attempt %d\n", attempt+1)
                break
            }
            fmt.Printf("Ping attempt %d failed: %v\n", attempt+1, err)
        }
        
        connected++
    }

    if connected == 0 {
        return fmt.Errorf("failed to connect to any bootstrap peers")
    }

    fmt.Printf("Connected to %d/%d bootstrap peers\n", connected, len(bootstrapPeers))

    // Important: Perform lookup to populate routing table
    fmt.Println("Performing network discovery...")
    nodeID := n.nodeIDManager.GetNodeID()
    contacts, err := n.FindNode(nodeID)
    if err != nil {
        fmt.Printf("Network discovery failed: %v\n", err)
        return err
    }
    
    fmt.Printf("Bootstrap complete: Found %d total contacts\n", len(contacts))
    return nil
}

func (n *Node) FindNode(target []byte) ([]Contact, error) {
    shortlist := n.routing.FindClosest(target, Alpha)
    if len(shortlist) == 0 {
        return nil, fmt.Errorf("no contacts in routing table")
    }

    queried := make(map[peer.ID]bool)
    var allContacts []Contact
    var mutex sync.Mutex

    // Iterative lookup with alpha parallelism
    for len(shortlist) > 0 && len(allContacts) < BucketSize {
        var wg sync.WaitGroup
        queries := 0

        for i := 0; i < len(shortlist) && queries < Alpha; i++ {
            contact := shortlist[i]
            if queried[contact.ID] {
                continue
            }
            queried[contact.ID] = true
            queries++

            wg.Add(1)
            go func(c Contact) {
                defer wg.Done()
                
                queryCtx, cancel := context.WithTimeout(n.ctx, 30*time.Second)
                defer cancel()
                
                msg := &Message{Type: FIND_NODE, Key: target}
                response, err := n.protocol.SendMessage(queryCtx, c.ID, msg)
                if err != nil {
                    return
                }

                if response.Type == FIND_NODE_RESPONSE {
                    mutex.Lock()
                    for _, peerInfo := range response.Peers {
                        newContact := Contact{
                            ID:       peerInfo.ID,
                            Addrs:    peerInfo.Addrs,
                            LastSeen: time.Now(),
                            NodeID:   GenerateNodeID(peerInfo.ID),
                        }
                        if newContact.IsValid() {
                            allContacts = append(allContacts, newContact)
                            n.routing.AddContact(newContact.ID, newContact.Addrs)
                        }
                    }
                    mutex.Unlock()
                }
            }(contact)
        }

        wg.Wait()

        // Update shortlist with closest unqueried contacts
        closest := n.routing.FindClosest(target, BucketSize)
        shortlist = nil
        for _, contact := range closest {
            if !queried[contact.ID] && len(shortlist) < Alpha {
                shortlist = append(shortlist, contact)
            }
        }
    }

    return n.routing.FindClosest(target, BucketSize), nil
}

func (n *Node) Store(key, value []byte) error {
    keyHash := sha1.Sum(key)
    contacts, err := n.FindNode(keyHash[:])
    if err != nil {
        return fmt.Errorf("failed to find nodes for storage: %w", err)
    }

    if len(contacts) == 0 {
        return fmt.Errorf("no nodes available for storage")
    }

    var wg sync.WaitGroup
    successCount := 0
    var mutex sync.Mutex

    // Store on closest nodes
    for _, contact := range contacts {
        wg.Add(1)
        go func(c Contact) {
            defer wg.Done()
            
            storeCtx, cancel := context.WithTimeout(n.ctx, 30*time.Second)
            defer cancel()
            
            msg := &Message{
                Type:  STORE,
                Key:   keyHash[:],
                Value: value,
            }
            
            _, err := n.protocol.SendMessage(storeCtx, c.ID, msg)
            if err == nil {
                mutex.Lock()
                successCount++
                mutex.Unlock()
            }
        }(contact)
    }

    wg.Wait()

    if successCount == 0 {
        return fmt.Errorf("failed to store value on any node")
    }

    fmt.Printf("Successfully stored value on %d/%d nodes\n", successCount, len(contacts))
    return nil
}

func (n *Node) FindValue(key []byte) ([]byte, error) {
    keyHash := sha1.Sum(key)
    shortlist := n.routing.FindClosest(keyHash[:], Alpha)
    
    if len(shortlist) == 0 {
        return nil, fmt.Errorf("no contacts in routing table")
    }

    queried := make(map[peer.ID]bool)

    // Iterative value lookup
    for len(shortlist) > 0 {
        var wg sync.WaitGroup
        var foundValue []byte
        var mutex sync.Mutex

        for _, contact := range shortlist {
            if queried[contact.ID] {
                continue
            }
            queried[contact.ID] = true

            wg.Add(1)
            go func(c Contact) {
                defer wg.Done()
                
                queryCtx, cancel := context.WithTimeout(n.ctx, 30*time.Second)
                defer cancel()
                
                msg := &Message{Type: FIND_VALUE, Key: keyHash[:]}
                response, err := n.protocol.SendMessage(queryCtx, c.ID, msg)
                if err != nil {
                    return
                }

                mutex.Lock()
                defer mutex.Unlock()
                
                if response.Type == FIND_VALUE_RESPONSE && response.Found && foundValue == nil {
                    foundValue = response.Value
                } else if response.Type == FIND_VALUE_RESPONSE && !response.Found {
                    // Add returned contacts to routing table
                    for _, peerInfo := range response.Peers {
                        n.routing.AddContact(peerInfo.ID, peerInfo.Addrs)
                    }
                }
            }(contact)
        }

        wg.Wait()

        if foundValue != nil {
            return foundValue, nil
        }

        // Update shortlist with closest unqueried contacts
        contacts := n.routing.FindClosest(keyHash[:], BucketSize)
        shortlist = nil
        for _, contact := range contacts {
            if !queried[contact.ID] && len(shortlist) < Alpha {
                shortlist = append(shortlist, contact)
            }
        }
    }

    return nil, fmt.Errorf("value not found in network")
}

func (n *Node) Ping(ctx context.Context, peerID peer.ID) error {
    // Check if we're connected at libp2p level first
    if len(n.host.Network().ConnsToPeer(peerID)) == 0 {
        return fmt.Errorf("not connected to peer %s", peerID)
    }
    
    // Create ping message
    msg := &Message{Type: PING}
    
    // Use a shorter timeout for ping
    pingCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
    defer cancel()
    
    start := time.Now()
    response, err := n.protocol.SendMessage(pingCtx, peerID, msg)
    latency := time.Since(start)
    
    if err != nil {
        return fmt.Errorf("ping failed: %w", err)
    }
    
    if response.Type != PONG {
        return fmt.Errorf("expected PONG, got %s", response.Type)
    }
    
    fmt.Printf("PING %s: time=%v\n", peerID, latency)
    return nil
}

// Public methods for monitoring and debugging
func (n *Node) GetPeerCount() int {
    return n.routing.GetPeerCount()
}

func (n *Node) GetNodeID() string {
    return n.nodeIDManager.GetNodeIDString()
}

func (n *Node) GetStorageInfo() (int, []string) {
    count := n.protocol.GetStorageCount()
    keys := n.protocol.GetStoredKeys()
    return count, keys
}

func (n *Node) GetBucketInfo() map[int]int {
    return n.routing.GetBucketInfo()
}

func (n *Node) GetAllContacts() []Contact {
    return n.routing.GetAllContacts()
}

func (n *Node) IsRunning() bool {
    n.mutex.RLock()
    defer n.mutex.RUnlock()
    return n.isRunning
}

func (n *Node) loadPersistedContacts() error {
    contacts, err := n.protocol.dataStore.LoadContacts()
    if err != nil {
        return err
    }
    
    loaded := 0
    for _, contact := range contacts {
        if n.routing.AddContact(contact.ID, contact.Addrs) {
            loaded++
        }
    }
    
    fmt.Printf("Loaded %d contacts from GORM database\n", loaded)
    return nil
}

func (n *Node) saveAllContacts() {
    contacts := n.routing.GetAllContacts()
    saved := 0
    
    for _, contact := range contacts {
        if err := n.protocol.dataStore.SaveContact(contact); err == nil {
            saved++
        }
    }
    
    fmt.Printf("Saved %d contacts to database\n", saved)
}

func (n *Node) GetDatabaseStats() map[string]interface{} {
    kvCount, contactCount, _ := n.protocol.dataStore.GetStorageInfo()
    stats, _ := n.protocol.dataStore.GetStats()
    sent, recv := n.protocol.GetMessageStats()
    
    result := map[string]interface{}{
        "stored_keys":      kvCount,
        "stored_contacts":  contactCount,
        "messages_sent":    sent,
        "messages_received": recv,
    }
    
    if stats != nil {
        result["last_updated"] = stats.LastUpdated.Format("2006-01-02 15:04:05")
    }
    
    return result
}
