package kademlia

import (
    "context"
    "crypto/sha256"
    "fmt"
    "sync"
    "time"

    "github.com/libp2p/go-libp2p/core/host"
    "github.com/libp2p/go-libp2p/core/peer"
)

// KademliaNode represents a Kademlia DHT node
type KademliaNode struct {
    host            host.Host
    protocolHandler *KademliaProtocolHandler
    ctx             context.Context
    cancel          context.CancelFunc
    bootstrapPeers  []peer.ID
    refreshInterval time.Duration
    mutex           sync.RWMutex
}

// NewKademliaNode creates a new Kademlia node
func NewKademliaNode(h host.Host, bootstrapPeers []peer.ID) *KademliaNode {
    ctx, cancel := context.WithCancel(context.Background())
    
    node := &KademliaNode{
        host:            h,
        protocolHandler: NewKademliaProtocolHandler(h),
        ctx:             ctx,
        cancel:          cancel,
        bootstrapPeers:  bootstrapPeers,
        refreshInterval: 1 * time.Hour, // Refresh routing table every hour
    }

    return node
}

// Start starts the Kademlia node
func (kn *KademliaNode) Start() error {
    fmt.Printf("Starting Kademlia node with ID: %s\n", kn.host.ID())

    // Bootstrap the node
    if err := kn.bootstrap(); err != nil {
        return fmt.Errorf("bootstrap failed: %w", err)
    }

    // Start periodic refresh
    go kn.periodicRefresh()

    return nil
}

// Stop stops the Kademlia node
func (kn *KademliaNode) Stop() {
    kn.cancel()
}

// bootstrap connects to bootstrap peers and populates routing table
func (kn *KademliaNode) bootstrap() error {
    if len(kn.bootstrapPeers) == 0 {
        fmt.Println("No bootstrap peers provided")
        return nil
    }

    fmt.Printf("Bootstrapping with %d peers\n", len(kn.bootstrapPeers))

    for _, peerID := range kn.bootstrapPeers {
        // Connect to bootstrap peer
        err := kn.host.Connect(kn.ctx, peer.AddrInfo{ID: peerID})
        if err != nil {
            fmt.Printf("Failed to connect to bootstrap peer %s: %v\n", peerID, err)
            continue
        }

        // Ping the peer
        msg := &KademliaMessage{Type: PING}
        _, err = kn.protocolHandler.SendMessage(kn.ctx, peerID, msg)
        if err != nil {
            fmt.Printf("Failed to ping bootstrap peer %s: %v\n", peerID, err)
            continue
        }

        fmt.Printf("Successfully connected to bootstrap peer: %s\n", peerID)
    }

    // Perform node lookup for our own ID to populate routing table
    localIDBytes := sha256.Sum256([]byte(kn.host.ID()))
    _, err := kn.FindNode(localIDBytes[:])
    return err
}

// FindNode performs iterative node lookup
func (kn *KademliaNode) FindNode(target []byte) ([]Contact, error) {
    shortlist := kn.protocolHandler.routingTable.FindClosestContacts(target, Alpha)
    
    if len(shortlist) == 0 {
        return nil, fmt.Errorf("no contacts in routing table")
    }

    queried := make(map[peer.ID]bool)
    var allContacts []Contact
    var mutex sync.Mutex

    // Iterative lookup
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

                msg := &KademliaMessage{
                    Type: FIND_NODE,
                    Key:  target,
                }

                response, err := kn.protocolHandler.SendMessage(kn.ctx, c.ID, msg)
                if err != nil {
                    return
                }

                if response.Type == FIND_NODE_RESPONSE {
                    mutex.Lock()
                    for _, peerInfo := range response.Peers {
                        contact := Contact{
                            ID:       peerInfo.ID,
                            Addrs:    peerInfo.Addrs,
                            Distance: XOR([]byte(peerInfo.ID), target),
                            LastSeen: time.Now(),
                        }
                        allContacts = append(allContacts, contact)
                        kn.protocolHandler.routingTable.AddContact(contact.ID, contact.Addrs)
                    }
                    mutex.Unlock()
                }
            }(contact)
        }

        wg.Wait()

        // Update shortlist with closest unqueried contacts
        allUnqueried := GetKClosest(allContacts, target, BucketSize)
        shortlist = nil
        for _, contact := range allUnqueried {
            if !queried[contact.ID] {
                shortlist = append(shortlist, contact)
            }
        }
    }

    return GetKClosest(allContacts, target, BucketSize), nil
}

// Store stores a key-value pair in the DHT
func (kn *KademliaNode) Store(key, value []byte) error {
    keyHash := sha256.Sum256(key)
    contacts, err := kn.FindNode(keyHash[:])
    if err != nil {
        return err
    }

    var wg sync.WaitGroup
    successCount := 0
    var mutex sync.Mutex

    for _, contact := range contacts {
        wg.Add(1)
        go func(c Contact) {
            defer wg.Done()

            msg := &KademliaMessage{
                Type:  STORE,
                Key:   keyHash[:],
                Value: value,
            }

            _, err := kn.protocolHandler.SendMessage(kn.ctx, c.ID, msg)
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

    fmt.Printf("Successfully stored value on %d nodes\n", successCount)
    return nil
}

// FindValue finds a value in the DHT
func (kn *KademliaNode) FindValue(key []byte) ([]byte, error) {
    keyHash := sha256.Sum256(key)
    shortlist := kn.protocolHandler.routingTable.FindClosestContacts(keyHash[:], Alpha)
    
    if len(shortlist) == 0 {
        return nil, fmt.Errorf("no contacts in routing table")
    }

    queried := make(map[peer.ID]bool)

    // Iterative value lookup
    for len(shortlist) > 0 {
        var wg sync.WaitGroup
        var foundValue []byte
        var mutex sync.Mutex
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

                msg := &KademliaMessage{
                    Type: FIND_VALUE,
                    Key:  keyHash[:],
                }

                response, err := kn.protocolHandler.SendMessage(kn.ctx, c.ID, msg)
                if err != nil {
                    return
                }

                if response.Type == FIND_VALUE_RESPONSE {
                    if response.Found {
                        mutex.Lock()
                        if foundValue == nil {
                            foundValue = response.Value
                        }
                        mutex.Unlock()
                        return
                    }

                    // Update routing table with returned contacts
                    for _, peerInfo := range response.Peers {
                        kn.protocolHandler.routingTable.AddContact(peerInfo.ID, peerInfo.Addrs)
                    }
                }
            }(contact)
        }

        wg.Wait()

        if foundValue != nil {
            return foundValue, nil
        }

        // Update shortlist with closest unqueried contacts
        contacts := kn.protocolHandler.routingTable.FindClosestContacts(keyHash[:], BucketSize)
        shortlist = nil
        for _, contact := range contacts {
            if !queried[contact.ID] {
                shortlist = append(shortlist, contact)
            }
        }
    }

    return nil, fmt.Errorf("value not found")
}

// periodicRefresh performs periodic routing table maintenance
func (kn *KademliaNode) periodicRefresh() {
    ticker := time.NewTicker(kn.refreshInterval)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            kn.refresh()
        case <-kn.ctx.Done():
            return
        }
    }
}

// refresh performs routing table refresh
func (kn *KademliaNode) refresh() {
    fmt.Println("Performing routing table refresh...")
    
    // Refresh each bucket by performing random lookups
    localIDBytes := sha256.Sum256([]byte(kn.host.ID()))
    
    // Perform a lookup for our own ID to refresh contacts
    _, err := kn.FindNode(localIDBytes[:])
    if err != nil {
        fmt.Printf("Refresh lookup failed: %v\n", err)
    }
}

// GetPeerCount returns the number of peers in the routing table
func (kn *KademliaNode) GetPeerCount() int {
    count := 0
    for i := 0; i < KeySize*8; i++ {
        count += kn.protocolHandler.routingTable.buckets[i].Size()
    }
    return count
}
