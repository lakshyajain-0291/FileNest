package kademlia

import (
    "encoding/json"
    "time"
)

// KeyValue represents stored key-value pairs
type KeyValue struct {
    Key       string     `gorm:"primaryKey;size:256"`
    Value     []byte     `gorm:"type:blob"`
    CreatedAt time.Time  `gorm:"autoCreateTime"`
    ExpiresAt *time.Time `gorm:"index"`
}

// ContactRecord represents routing table contacts
type ContactRecord struct {
    PeerID    string    `gorm:"primaryKey;size:128"`
    NodeID    string    `gorm:"size:64;index"`
    Addrs     string    `gorm:"type:text"` // JSON-encoded []string
    LastSeen  time.Time `gorm:"autoUpdateTime;index"`
    IsActive  bool      `gorm:"default:true;index"`
}

// NodeStats represents node statistics
type NodeStats struct {
    ID            uint      `gorm:"primaryKey"`
    MessagesSent  int64     `gorm:"default:0"`
    MessagesRecv  int64     `gorm:"default:0"`
    PeerCount     int       `gorm:"default:0"`
    StoredKeys    int       `gorm:"default:0"`
    LastUpdated   time.Time `gorm:"autoUpdateTime"`
}

// Helper methods for ContactRecord
func (cr *ContactRecord) GetAddrs() []string {
    var addrs []string
    json.Unmarshal([]byte(cr.Addrs), &addrs)
    return addrs
}

func (cr *ContactRecord) SetAddrs(addrs []string) {
    data, _ := json.Marshal(addrs)
    cr.Addrs = string(data)
}
