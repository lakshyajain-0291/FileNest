package kademlia

import (
    "bufio"
    "context"
    "fmt"
    "io"
    "sync"
    "time"

    "github.com/google/uuid"
    "github.com/libp2p/go-libp2p/core/host"
    "github.com/libp2p/go-libp2p/core/network"
    "github.com/libp2p/go-libp2p/core/peer"
    "github.com/libp2p/go-libp2p/core/protocol"
)

const ProtocolID = "/kademlia/1.0.0"

type ProtocolHandler struct {
    host         host.Host
    routingTable *RoutingTable
    dataStore    map[string][]byte
    pending      map[string]chan *Message
    mutex        sync.RWMutex
}

func NewProtocolHandler(host host.Host, routingTable *RoutingTable) *ProtocolHandler {
    ph := &ProtocolHandler{
        host:         host,
        routingTable: routingTable,
        dataStore:    make(map[string][]byte),
        pending:      make(map[string]chan *Message),
    }
    
    host.SetStreamHandler(protocol.ID(ProtocolID), ph.handleStream)
    return ph
}

func (ph *ProtocolHandler) handleStream(s network.Stream) {
    defer s.Close()
    reader := bufio.NewReader(s)
    writer := bufio.NewWriter(s)

    for {
        msgBytes, err := reader.ReadBytes('\n')
        if err != nil {
            if err != io.EOF {
                fmt.Printf("Read error: %v\n", err)
            }
            return
        }

        msg, err := DeserializeMessage(msgBytes[:len(msgBytes)-1])
        if err != nil {
            continue
        }

        response := ph.handleMessage(msg, s.Conn().RemotePeer())
        if response != nil {
            responseBytes, _ := response.Serialize()
            writer.Write(append(responseBytes, '\n'))
            writer.Flush()
        }
    }
}

func (ph *ProtocolHandler) handleMessage(msg *Message, remotePeer peer.ID) *Message {
    // Update routing table
    ph.routingTable.AddContact(remotePeer, []string{})

    switch msg.Type {
    case PING:
        return &Message{Type: PONG, ID: msg.ID, Timestamp: time.Now().Unix()}

    case PONG:
        ph.notifyPending(msg)
        return nil

    case FIND_NODE:
        contacts := ph.routingTable.FindClosest(msg.Key, BucketSize)
        peers := make([]PeerInfo, len(contacts))
        for i, c := range contacts {
            peers[i] = PeerInfo{ID: c.ID, Addrs: c.Addrs}
        }
        return &Message{Type: FIND_NODE_RESPONSE, ID: msg.ID, Peers: peers, Timestamp: time.Now().Unix()}

    case STORE:
        ph.mutex.Lock()
        ph.dataStore[string(msg.Key)] = msg.Value
        ph.mutex.Unlock()
        return &Message{Type: STORE_RESPONSE, ID: msg.ID, Timestamp: time.Now().Unix()}

    case FIND_VALUE:
        ph.mutex.RLock()
        value, exists := ph.dataStore[string(msg.Key)]
        ph.mutex.RUnlock()

        if exists {
            return &Message{Type: FIND_VALUE_RESPONSE, ID: msg.ID, Value: value, Found: true, Timestamp: time.Now().Unix()}
        } else {
            contacts := ph.routingTable.FindClosest(msg.Key, BucketSize)
            peers := make([]PeerInfo, len(contacts))
            for i, c := range contacts {
                peers[i] = PeerInfo{ID: c.ID, Addrs: c.Addrs}
            }
            return &Message{Type: FIND_VALUE_RESPONSE, ID: msg.ID, Peers: peers, Found: false, Timestamp: time.Now().Unix()}
        }

    default:
        ph.notifyPending(msg)
        return nil
    }
}

func (ph *ProtocolHandler) SendMessage(ctx context.Context, peerID peer.ID, msg *Message) (*Message, error) {
    msg.ID = uuid.New().String()
    msg.Timestamp = time.Now().Unix()

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

    stream, err := ph.host.NewStream(ctx, peerID, protocol.ID(ProtocolID))
    if err != nil {
        return nil, err
    }
    defer stream.Close()

    msgBytes, _ := msg.Serialize()
    writer := bufio.NewWriter(stream)
    writer.Write(append(msgBytes, '\n'))
    writer.Flush()

    select {
    case response := <-responseChan:
        return response, nil
    case <-time.After(30 * time.Second):
        return nil, fmt.Errorf("timeout")
    }
}

func (ph *ProtocolHandler) notifyPending(msg *Message) {
    ph.mutex.RLock()
    if ch, exists := ph.pending[msg.ID]; exists {
        select {
        case ch <- msg:
        default:
        }
    }
    ph.mutex.RUnlock()
}
