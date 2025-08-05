package kademlia

import (
	"bufio"
	"context"
	"crypto/sha256"
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

const (
	KademliaProtocol = "/kademlia/1.0.0"
	RequestTimeout   = 30 * time.Second
)

// KademliaProtocolHandler handles the Kademlia protocol
type KademliaProtocolHandler struct {
	host            host.Host
	routingTable    *RoutingTable
	dataStore       map[string][]byte
	pendingRequests map[string]chan *KademliaMessage
	mutex           sync.RWMutex
}

// NewKademliaProtocolHandler creates a new protocol handler
func NewKademliaProtocolHandler(h host.Host) *KademliaProtocolHandler {
	localIDBytes := sha256.Sum256([]byte(h.ID()))

	handler := &KademliaProtocolHandler{
		host:            h,
		routingTable:    NewRoutingTable(localIDBytes[:]),
		dataStore:       make(map[string][]byte),
		pendingRequests: make(map[string]chan *KademliaMessage),
	}

	h.SetStreamHandler(protocol.ID(KademliaProtocol), handler.handleStream)
	return handler
}

// handleStream handles incoming streams
func (kph *KademliaProtocolHandler) handleStream(s network.Stream) {
	defer s.Close()

	reader := bufio.NewReader(s)
	writer := bufio.NewWriter(s)

	for {
		// Read message
		msgBytes, err := reader.ReadBytes('\n')
		if err != nil {
			if err != io.EOF {
				fmt.Printf("Error reading message: %v\n", err)
			}
			return
		}

		// Deserialize message
		msg, err := DeserializeMessage(msgBytes[:len(msgBytes)-1]) // Remove newline
		if err != nil {
			fmt.Printf("Error deserializing message: %v\n", err)
			continue
		}

		// Handle message
		response := kph.handleMessage(msg, s.Conn().RemotePeer())
		if response != nil {
			// Serialize and send response
			responseBytes, err := response.Serialize()
			if err != nil {
				fmt.Printf("Error serializing response: %v\n", err)
				continue
			}

			_, err = writer.Write(append(responseBytes, '\n'))
			if err != nil {
				fmt.Printf("Error writing response: %v\n", err)
				return
			}
			writer.Flush()
		}
	}
}

// handleMessage processes incoming Kademlia messages
func (kph *KademliaProtocolHandler) handleMessage(msg *KademliaMessage, remotePeer peer.ID) *KademliaMessage {
	// Update routing table with remote peer
	kph.routingTable.AddContact(remotePeer, []string{}) // In real implementation, extract addresses

	switch msg.Type {
	case PING:
		return &KademliaMessage{
			Type:      PONG,
			ID:        msg.ID,
			Timestamp: time.Now().Unix(),
		}

	case FIND_NODE:
		contacts := kph.routingTable.FindClosestContacts(msg.Key, BucketSize)
		peers := make([]PeerInfo, len(contacts))
		for i, contact := range contacts {
			peers[i] = PeerInfo{
				ID:       contact.ID,
				Addrs:    contact.Addrs,
				Distance: contact.Distance,
			}
		}
		return &KademliaMessage{
			Type:      FIND_NODE_RESPONSE,
			ID:        msg.ID,
			Peers:     peers,
			Timestamp: time.Now().Unix(),
		}

	case STORE:
		kph.mutex.Lock()
		kph.dataStore[string(msg.Key)] = msg.Value
		kph.mutex.Unlock()
		return &KademliaMessage{
			Type:      STORE_RESPONSE,
			ID:        msg.ID,
			Timestamp: time.Now().Unix(),
		}

	case FIND_VALUE:
		kph.mutex.RLock()
		value, exists := kph.dataStore[string(msg.Key)]
		kph.mutex.RUnlock()

		if exists {
			return &KademliaMessage{
				Type:      FIND_VALUE_RESPONSE,
				ID:        msg.ID,
				Value:     value,
				Found:     true,
				Timestamp: time.Now().Unix(),
			}
		} else {
			// Return closest nodes
			contacts := kph.routingTable.FindClosestContacts(msg.Key, BucketSize)
			peers := make([]PeerInfo, len(contacts))
			for i, contact := range contacts {
				peers[i] = PeerInfo{
					ID:       contact.ID,
					Addrs:    contact.Addrs,
					Distance: contact.Distance,
				}
			}
			return &KademliaMessage{
				Type:      FIND_VALUE_RESPONSE,
				ID:        msg.ID,
				Peers:     peers,
				Found:     false,
				Timestamp: time.Now().Unix(),
			}
		}

	case PONG, FIND_NODE_RESPONSE, STORE_RESPONSE, FIND_VALUE_RESPONSE:
		// Handle responses by notifying waiting goroutines
		kph.mutex.RLock()
		if ch, exists := kph.pendingRequests[msg.ID]; exists {
			select {
			case ch <- msg:
			default:
			}
		}
		kph.mutex.RUnlock()
		return nil
	}

	return nil
}

// SendMessage sends a message to a peer and waits for response
func (kph *KademliaProtocolHandler) SendMessage(ctx context.Context, peerID peer.ID, msg *KademliaMessage) (*KademliaMessage, error) {
    // Generate unique ID for this request
    msg.ID = uuid.New().String()
    msg.Timestamp = time.Now().Unix()

    // Create response channel
    responseChan := make(chan *KademliaMessage, 1)
    kph.mutex.Lock()
    kph.pendingRequests[msg.ID] = responseChan
    kph.mutex.Unlock()

    defer func() {
        kph.mutex.Lock()
        delete(kph.pendingRequests, msg.ID)
        kph.mutex.Unlock()
        close(responseChan)
    }()

    // Open stream to peer
    stream, err := kph.host.NewStream(ctx, peerID, protocol.ID(KademliaProtocol))
    if err != nil {
        return nil, fmt.Errorf("failed to open stream: %w", err)
    }
    defer stream.Close()

    // Send message
    msgBytes, err := msg.Serialize()
    if err != nil {
        return nil, fmt.Errorf("failed to serialize message: %w", err)
    }

    writer := bufio.NewWriter(stream)
    _, err = writer.Write(append(msgBytes, '\n'))
    if err != nil {
        return nil, fmt.Errorf("failed to write message: %w", err)
    }
    err = writer.Flush()
    if err != nil {
        return nil, fmt.Errorf("failed to flush message: %w", err)
    }

    // Create timeout context
    timeoutCtx, cancel := context.WithTimeout(ctx, RequestTimeout)
    defer cancel()

    // Wait for response
    select {
    case response := <-responseChan:
        return response, nil
    case <-timeoutCtx.Done():
        return nil, fmt.Errorf("request timeout")
    }
}
