package kademlia

import (
    "encoding/json"
    "time"
    "github.com/libp2p/go-libp2p/core/peer"
)

const (
    BucketSize = 20
    KeySize    = 32
    Alpha      = 3
)

// Message types
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

// Message structure
type Message struct {
    Type      MessageType `json:"type"`
    ID        string      `json:"id"`
    Key       []byte      `json:"key,omitempty"`
    Value     []byte      `json:"value,omitempty"`
    Peers     []PeerInfo  `json:"peers,omitempty"`
    Found     bool        `json:"found,omitempty"`
    Timestamp int64       `json:"timestamp"`
}

type PeerInfo struct {
    ID    peer.ID  `json:"id"`
    Addrs []string `json:"addrs"`
}

type Contact struct {
    ID       peer.ID
    Addrs    []string
    LastSeen time.Time
}

// Serialize message
func (m *Message) Serialize() ([]byte, error) {
    return json.Marshal(m)
}

// Deserialize message
func DeserializeMessage(data []byte) (*Message, error) {
    var msg Message
    err := json.Unmarshal(data, &msg)
    return &msg, err
}
