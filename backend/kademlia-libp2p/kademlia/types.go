package kademlia

import (
    "encoding/json"
    "fmt"
    "time"
    "github.com/libp2p/go-libp2p/core/peer"
)

const (
    BucketSize = 20
    KeySize    = 20
    Alpha      = 3
    ProtocolID = "/kademlia/1.0.0"
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

func (mt MessageType) String() string {
    switch mt {
    case PING: return "PING"
    case PONG: return "PONG"
    case FIND_NODE: return "FIND_NODE"
    case FIND_NODE_RESPONSE: return "FIND_NODE_RESPONSE"
    case STORE: return "STORE"
    case STORE_RESPONSE: return "STORE_RESPONSE"
    case FIND_VALUE: return "FIND_VALUE"
    case FIND_VALUE_RESPONSE: return "FIND_VALUE_RESPONSE"
    default: return "UNKNOWN"
    }
}

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
    NodeID   []byte
}

// Validation
func (c *Contact) IsValid() bool {
    return c.ID != "" && len(c.NodeID) == KeySize
}

func (c *Contact) String() string {
    return fmt.Sprintf("Contact{ID: %s, Addrs: %v, LastSeen: %v}", 
        c.ID.String()[:12]+"...", c.Addrs, c.LastSeen.Format("15:04:05"))
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

// Validate message
func (m *Message) IsValid() bool {
    return m.ID != "" && m.Timestamp > 0
}
