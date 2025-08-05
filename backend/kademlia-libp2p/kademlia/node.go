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

type Node struct {
    host     host.Host
    nodeID   []byte
    routing  *RoutingTable
    protocol *ProtocolHandler
    ctx      context.Context
    cancel   context.CancelFunc
}

func NewNode(host host.Host, dataDir string) (*Node, error) {
    // Get persistent node ID
    nodeID, err := GetOrCreateNodeID(dataDir, host.ID())
    if err != nil {
        return nil, err
    }

    ctx, cancel := context.WithCancel(context.Background())
    routing := NewRoutingTable(nodeID)
    protocol := NewProtocolHandler(host, routing)

    return &Node{
        host:     host,
        nodeID:   nodeID,
        routing:  routing,
        protocol: protocol,
        ctx:      ctx,
        cancel:   cancel,
    }, nil
}

func (n *Node) Bootstrap(bootstrapPeers []peer.ID) error {
    for _, peerID := range bootstrapPeers {
        // Connect
        err := n.host.Connect(n.ctx, peer.AddrInfo{ID: peerID})
        if err != nil {
            fmt.Printf("Failed to connect to %s: %v\n", peerID, err)
            continue
        }

        // Ping
        msg := &Message{Type: PING}
        _, err = n.protocol.SendMessage(n.ctx, peerID, msg)
        if err != nil {
            fmt.Printf("Failed to ping %s: %v\n", peerID, err)
            continue
        }

        fmt.Printf("Connected to bootstrap peer: %s\n", peerID)
    }

    // Find nodes close to ourselves
    _, err := n.FindNode(n.nodeID)
    return err
}

func (n *Node) FindNode(target []byte) ([]Contact, error) {
    shortlist := n.routing.FindClosest(target, Alpha)
    if len(shortlist) == 0 {
        return nil, fmt.Errorf("no contacts")
    }

    queried := make(map[peer.ID]bool)
    var allContacts []Contact
    var mutex sync.Mutex

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
                msg := &Message{Type: FIND_NODE, Key: target}
                response, err := n.protocol.SendMessage(n.ctx, c.ID, msg)
                if err != nil {
                    return
                }

                if response.Type == FIND_NODE_RESPONSE {
                    mutex.Lock()
                    for _, peerInfo := range response.Peers {
                        contact := Contact{
                            ID:       peerInfo.ID,
                            Addrs:    peerInfo.Addrs,
                            LastSeen: time.Now(),
                        }
                        allContacts = append(allContacts, contact)
                        n.routing.AddContact(contact.ID, contact.Addrs)
                    }
                    mutex.Unlock()
                }
            }(contact)
        }

        wg.Wait()

        // Update shortlist
        closest := n.routing.FindClosest(target, BucketSize)
        shortlist = nil
        for _, contact := range closest {
            if !queried[contact.ID] {
                shortlist = append(shortlist, contact)
            }
        }
    }

    return n.routing.FindClosest(target, BucketSize), nil
}

func (n *Node) Store(key, value []byte) error {
    keyHash := sha256.Sum256(key)
    contacts, err := n.FindNode(keyHash[:])
    if err != nil {
        return err
    }

    var wg sync.WaitGroup
    for _, contact := range contacts {
        wg.Add(1)
        go func(c Contact) {
            defer wg.Done()
            msg := &Message{Type: STORE, Key: keyHash[:], Value: value}
            n.protocol.SendMessage(n.ctx, c.ID, msg)
        }(contact)
    }
    wg.Wait()

    fmt.Printf("Stored on %d nodes\n", len(contacts))
    return nil
}

func (n *Node) FindValue(key []byte) ([]byte, error) {
    keyHash := sha256.Sum256(key)
    shortlist := n.routing.FindClosest(keyHash[:], Alpha)
    queried := make(map[peer.ID]bool)

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
                msg := &Message{Type: FIND_VALUE, Key: keyHash[:]}
                response, err := n.protocol.SendMessage(n.ctx, c.ID, msg)
                if err != nil {
                    return
                }

                if response.Type == FIND_VALUE_RESPONSE && response.Found {
                    mutex.Lock()
                    if foundValue == nil {
                        foundValue = response.Value
                    }
                    mutex.Unlock()
                }
            }(contact)
        }

        wg.Wait()

        if foundValue != nil {
            return foundValue, nil
        }

        // Update shortlist
        contacts := n.routing.FindClosest(keyHash[:], BucketSize)
        shortlist = nil
        for _, contact := range contacts {
            if !queried[contact.ID] {
                shortlist = append(shortlist, contact)
            }
        }
    }

    return nil, fmt.Errorf("value not found")
}

func (n *Node) Ping(peerID peer.ID) error {
    msg := &Message{Type: PING}
    response, err := n.protocol.SendMessage(n.ctx, peerID, msg)
    if err != nil {
        return err
    }
    if response.Type != PONG {
        return fmt.Errorf("expected PONG")
    }
    return nil
}

func (n *Node) GetPeerCount() int {
    return n.routing.GetPeerCount()
}

func (n *Node) Stop() {
    n.cancel()
}
