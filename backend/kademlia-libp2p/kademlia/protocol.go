package kademlia

import (
    "bufio"
    "context"
    "fmt"
    "io"
    "strings"
    "sync"
    "sync/atomic"
    "time"

    "github.com/google/uuid"
    "github.com/libp2p/go-libp2p/core/host"
    "github.com/libp2p/go-libp2p/core/network"
    "github.com/libp2p/go-libp2p/core/peer"
    "github.com/libp2p/go-libp2p/core/protocol"
)

type ProtocolHandler struct {
    host         host.Host
    routingTable *RoutingTable
    dataStore    *DataStore
    pending      map[string]chan *Message
    mutex        sync.RWMutex
    nodeID       []byte
    
    // Statistics
    messagesSent int64
    messagesRecv int64
}

func NewProtocolHandler(host host.Host, routingTable *RoutingTable, nodeID []byte, dataDir string) (*ProtocolHandler, error) {
    dataStore, err := NewDataStore(dataDir)
    if err != nil {
        return nil, fmt.Errorf("failed to create datastore: %w", err)
    }

    ph := &ProtocolHandler{
        host:         host,
        routingTable: routingTable,
        dataStore:    dataStore,
        pending:      make(map[string]chan *Message),
        nodeID:       nodeID,
    }
    
    host.SetStreamHandler(protocol.ID(ProtocolID), ph.handleStream)
    return ph, nil
}

func (ph *ProtocolHandler) handleStream(s network.Stream) {
    defer s.Close()
    
    remotePeer := s.Conn().RemotePeer()
    reader := bufio.NewReader(s)
    writer := bufio.NewWriter(s)
    
    // Extract real addresses from connection
    addrs := ph.extractAddresses(s)
    
    // Add peer to routing table immediately upon connection
    ph.routingTable.AddContact(remotePeer, addrs)

    // Handle only ONE message per stream (important for ping)
    msgBytes, err := reader.ReadBytes('\n')
    if err != nil {
        if err != io.EOF {
            fmt.Printf("Stream read error from %s: %v\n", remotePeer, err)
        }
        return
    }

    msg, err := DeserializeMessage(msgBytes[:len(msgBytes)-1])
    if err != nil {
        fmt.Printf("Failed to deserialize message from %s: %v\n", remotePeer, err)
        return
    }

    if !msg.IsValid() {
        fmt.Printf("Invalid message from %s\n", remotePeer)
        return
    }

    fmt.Printf("Received %s from %s\n", msg.Type, remotePeer) // Debug

    response := ph.handleMessage(msg, remotePeer, addrs)
    if response != nil {
        fmt.Printf("Sending %s to %s\n", response.Type, remotePeer) // Debug
        
        responseBytes, err := response.Serialize()
        if err != nil {
            fmt.Printf("Failed to serialize response: %v\n", err)
            return
        }
        
        if _, err := writer.Write(append(responseBytes, '\n')); err != nil {
            fmt.Printf("Failed to write response: %v\n", err)
            return
        }
        if err := writer.Flush(); err != nil {
            fmt.Printf("Failed to flush response: %v\n", err)
            return
        }
        
        fmt.Printf("Successfully sent %s to %s\n", response.Type, remotePeer) // Debug
    }
}

func (ph *ProtocolHandler) extractAddresses(s network.Stream) []string {
    var addrs []string
    
    // Get remote multiaddr from the stream
    remoteAddr := s.Conn().RemoteMultiaddr()
    if remoteAddr != nil {
        // Extract the network part (without /p2p/peerID)
        addrStr := remoteAddr.String()
        if idx := strings.Index(addrStr, "/p2p/"); idx != -1 {
            addrStr = addrStr[:idx]
        }
        addrs = append(addrs, addrStr)
    }
    
    return addrs
}

func (ph *ProtocolHandler) handleMessage(msg *Message, remotePeer peer.ID, addrs []string) *Message {
    // Increment received messages counter
    atomic.AddInt64(&ph.messagesRecv, 1)
    
    // Always update routing table when we receive a message
    ph.routingTable.AddContact(remotePeer, addrs)

    switch msg.Type {
    case PING:
        fmt.Printf("Received PING from %s\n", remotePeer)
        return &Message{
            Type:      PONG,
            ID:        msg.ID,
            Timestamp: time.Now().Unix(),
        }

    case PONG:
        fmt.Printf("Received PONG from %s\n", remotePeer)
        ph.notifyPending(msg)
        return nil

    case FIND_NODE:
        contacts := ph.routingTable.FindClosest(msg.Key, BucketSize)
        peers := make([]PeerInfo, 0, len(contacts))
        
        for _, contact := range contacts {
            // Don't return the requesting peer
            if contact.ID != remotePeer {
                peers = append(peers, PeerInfo{
                    ID:    contact.ID,
                    Addrs: contact.Addrs,
                })
            }
        }
        
        return &Message{
            Type:      FIND_NODE_RESPONSE,
            ID:        msg.ID,
            Peers:     peers,
            Timestamp: time.Now().Unix(),
        }

    case STORE:
        // Store with 24-hour TTL
        err := ph.dataStore.Put(string(msg.Key), msg.Value, 24*time.Hour)
        if err != nil {
            fmt.Printf("Failed to store key: %v\n", err)
        } else {
            fmt.Printf("Stored key: %x (value length: %d)\n", msg.Key, len(msg.Value))
        }
        
        return &Message{
            Type:      STORE_RESPONSE,
            ID:        msg.ID,
            Timestamp: time.Now().Unix(),
        }

    case FIND_VALUE:
        value, err := ph.dataStore.Get(string(msg.Key))
        if err == nil {
            return &Message{
                Type:      FIND_VALUE_RESPONSE,
                ID:        msg.ID,
                Value:     value,
                Found:     true,
                Timestamp: time.Now().Unix(),
            }
        } else {
            // Return closest nodes
            contacts := ph.routingTable.FindClosest(msg.Key, BucketSize)
            peers := make([]PeerInfo, 0, len(contacts))
            
            for _, contact := range contacts {
                if contact.ID != remotePeer {
                    peers = append(peers, PeerInfo{
                        ID:    contact.ID,
                        Addrs: contact.Addrs,
                    })
                }
            }
            
            return &Message{
                Type:      FIND_VALUE_RESPONSE,
                ID:        msg.ID,
                Peers:     peers,
                Found:     false,
                Timestamp: time.Now().Unix(),
            }
        }

    case FIND_NODE_RESPONSE, STORE_RESPONSE, FIND_VALUE_RESPONSE:
        // Handle responses by notifying waiting goroutines
        ph.notifyPending(msg)
        return nil

    default:
        fmt.Printf("Unknown message type: %s from %s\n", msg.Type, remotePeer)
        return nil
    }
}

func (ph *ProtocolHandler) SendMessage(ctx context.Context, peerID peer.ID, msg *Message) (*Message, error) {
    msg.ID = uuid.New().String()
    msg.Timestamp = time.Now().Unix()

    fmt.Printf("Sending %s to %s (ID: %s)\n", msg.Type, peerID, msg.ID) // Debug

    // Increment sent messages counter
    atomic.AddInt64(&ph.messagesSent, 1)

    responseChan := make(chan *Message, 1)
    ph.mutex.Lock()
    ph.pending[msg.ID] = responseChan
    ph.mutex.Unlock()

    defer func() {
        ph.mutex.Lock()
        delete(ph.pending, msg.ID)
        ph.mutex.Unlock()
        close(responseChan)
    }()

    // Open stream to peer
    stream, err := ph.host.NewStream(ctx, peerID, protocol.ID(ProtocolID))
    if err != nil {
        return nil, fmt.Errorf("failed to open stream to %s: %w", peerID, err)
    }
    defer stream.Close()

    // Send message
    msgBytes, err := msg.Serialize()
    if err != nil {
        return nil, fmt.Errorf("failed to serialize message: %w", err)
    }

    writer := bufio.NewWriter(stream)
    if _, err := writer.Write(append(msgBytes, '\n')); err != nil {
        return nil, fmt.Errorf("failed to write message: %w", err)
    }
    if err := writer.Flush(); err != nil {
        return nil, fmt.Errorf("failed to flush message: %w", err)
    }

    fmt.Printf("Message sent, waiting for response...\n") // Debug

    // Read response directly from the same stream
    reader := bufio.NewReader(stream)
    responseBytes, err := reader.ReadBytes('\n')
    if err != nil {
        return nil, fmt.Errorf("failed to read response: %w", err)
    }

    response, err := DeserializeMessage(responseBytes[:len(responseBytes)-1])
    if err != nil {
        return nil, fmt.Errorf("failed to deserialize response: %w", err)
    }

    fmt.Printf("Received %s response (ID: %s)\n", response.Type, response.ID) // Debug

    // Validate response ID matches request ID
    if response.ID != msg.ID {
        return nil, fmt.Errorf("response ID mismatch: expected %s, got %s", msg.ID, response.ID)
    }

    return response, nil
}

func (ph *ProtocolHandler) notifyPending(msg *Message) {
    ph.mutex.RLock()
    defer ph.mutex.RUnlock()
    
    if ch, exists := ph.pending[msg.ID]; exists {
        select {
        case ch <- msg:
        default:
        }
    }
}

func (ph *ProtocolHandler) GetStoredKeys() []string {
    return ph.dataStore.GetStoredKeys()
}

func (ph *ProtocolHandler) GetStorageCount() int {
    kvCount, _, _ := ph.dataStore.GetStorageInfo()
    return int(kvCount)
}

func (ph *ProtocolHandler) GetMessageStats() (int64, int64) {
    return atomic.LoadInt64(&ph.messagesSent), atomic.LoadInt64(&ph.messagesRecv)
}
