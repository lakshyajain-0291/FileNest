package kademlia

import (
    "encoding/json"
    "github.com/libp2p/go-libp2p/core/peer"
)

// Message types for Kademlia protocol
type MessageType int

const (
    PING MessageType = iota
    PONG
    FIND_NODE
    FIND_NODE_RESPONSE
    STORE
    STORE_RESPONSE
    FIND_VALUE
    FIND_VALUE_RESPONSE
)

// KademliaMessage represents a protocol message
type KademliaMessage struct {
    Type      MessageType `json:"type"`
    ID        string      `json:"id"`
    Key       []byte      `json:"key,omitempty"`
    Value     []byte      `json:"value,omitempty"`
    Peers     []PeerInfo  `json:"peers,omitempty"`
    Found     bool        `json:"found,omitempty"`
    Timestamp int64       `json:"timestamp"`
}

// PeerInfo represents peer contact information
type PeerInfo struct {
    ID       peer.ID    `json:"id"`
    Addrs    []string   `json:"addrs"`
    Distance []byte     `json:"distance"`
}

// Serialize message to JSON
func (m *KademliaMessage) Serialize() ([]byte, error) {
    return json.Marshal(m)
}

// Deserialize message from JSON
func DeserializeMessage(data []byte) (*KademliaMessage, error) {
    var msg KademliaMessage
    err := json.Unmarshal(data, &msg)
    return &msg, err
}
