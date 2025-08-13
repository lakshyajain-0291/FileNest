package kademlia

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"

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

// Update your handleMessage method to ensure PING/PONG works properly
func (ph *ProtocolHandler) handleMessage(msg *Message, remotePeer peer.ID, addrs []string) *Message {
	// ALWAYS add the peer to routing table when we receive any message
	ph.routingTable.AddContact(remotePeer, addrs)

	log.Printf("Received %s message from %s", msg.Type, remotePeer)

	switch msg.Type {
	case PING:
		log.Printf("Responding to PING from %s", remotePeer)
		// Send PONG response
		return &Message{
			Type:      PONG,
			ID:        msg.ID, // Same ID as the request
			Timestamp: time.Now().Unix(),
		}

	case PONG:
		log.Printf("Received PONG from %s", remotePeer)
		// PONG is handled by the response mechanism in SendMessage
		// Just ensure the peer is in our routing table (already done above)
		return nil

	case FIND_NODE:
		// Find k closest nodes to target
		contacts := ph.routingTable.FindClosest(msg.Key, BucketSize)
		peers := make([]PeerInfo, 0, len(contacts))

		for _, contact := range contacts {
			if contact.ID != remotePeer { // Don't send the requester back to themselves
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

	case FIND_VALUE:
		// Try to get value locally first
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
			// Value not found locally, return closest nodes
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

	// case STORE:
	// 	// Store the key-value pair
	// 	err := ph.dataStore.Store(string(msg.Key), msg.Value)
	// 	if err != nil {
	// 		log.Printf("Failed to store key-value pair: %v", err)
	// 	}

	// 	return &Message{
	// 		Type:      STORE_RESPONSE,
	// 		ID:        msg.ID,
	// 		Timestamp: time.Now().Unix(),
	// 	}

	case FIND_NODE_RESPONSE, FIND_VALUE_RESPONSE, STORE_RESPONSE:
		// These are responses - they should not be handled here
		// They're handled by the response mechanism in SendMessage
		return nil

	default:
		log.Printf("Unknown message type: %v", msg.Type)
		return nil
	}
}

// Ensure your SendMessage method has proper response handling
func (ph *ProtocolHandler) SendMessage(ctx context.Context, peerID peer.ID, msg *Message) (*Message, error) {
	stream, err := ph.host.NewStream(ctx, peerID, protocol.ID(ProtocolID))
	if err != nil {
		return nil, fmt.Errorf("failed to open stream: %w", err)
	}
	defer stream.Close()

	// Set stream deadlines
	stream.SetDeadline(time.Now().Add(30 * time.Second))

	rw := bufio.NewReadWriter(bufio.NewReader(stream), bufio.NewWriter(stream))

	// Send message
	msgBytes, err := json.Marshal(msg)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal message: %w", err)
	}

	_, err = rw.Write(append(msgBytes, '\n'))
	if err != nil {
		return nil, fmt.Errorf("failed to write message: %w", err)
	}

	err = rw.Flush()
	if err != nil {
		return nil, fmt.Errorf("failed to flush message: %w", err)
	}

	// For PING messages, wait for PONG response
	if msg.Type == PING {
		responseBytes, err := rw.ReadBytes('\n')
		if err != nil {
			return nil, fmt.Errorf("failed to read response: %w", err)
		}

		var response Message
		err = json.Unmarshal(responseBytes, &response)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal response: %w", err)
		}

		return &response, nil
	}

	// For other message types that expect responses
	if msg.Type == FIND_NODE || msg.Type == FIND_VALUE {
		responseBytes, err := rw.ReadBytes('\n')
		if err != nil {
			return nil, fmt.Errorf("failed to read response: %w", err)
		}

		var response Message
		err = json.Unmarshal(responseBytes, &response)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal response: %w", err)
		}

		return &response, nil
	}

	return nil, nil // No response expected for STORE messages
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

// // In your Ping method in node.go
// func (n *Node) Ping(peerID peer.ID) error {
//     msg := &Message{
//         Type:      PING,
//         ID:        uuid.New().String(), // Direct UUID generation
//         Timestamp: time.Now().Unix(),
//     }

//     ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
//     defer cancel()

//     response, err := n.protocol.SendMessage(ctx, peerID, msg)
//     if err != nil {
//         return fmt.Errorf("ping failed: %w", err)
//     }

//     if response.Type != PONG {
//         return fmt.Errorf("unexpected response type: %v", response.Type)
//     }

//     return nil
// }

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
