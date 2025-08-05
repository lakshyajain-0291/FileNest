package kademlia

import (
    "sort"
    "sync"
    "time"
    "github.com/libp2p/go-libp2p/core/peer"
)

type RoutingTable struct {
    nodeID   []byte
    buckets  [][]Contact
    mutex    sync.RWMutex
}

func NewRoutingTable(nodeID []byte) *RoutingTable {
    return &RoutingTable{
        nodeID:  nodeID,
        buckets: make([][]Contact, KeySize*8),
    }
}

func (rt *RoutingTable) AddContact(peerID peer.ID, addrs []string) {
    if peerID == "" {
        return
    }
    
    distance := XORDistance(rt.nodeID, []byte(peerID))
    bucketIndex := getBucketIndex(distance)
    
    contact := Contact{
        ID:       peerID,
        Addrs:    addrs,
        LastSeen: time.Now(),
    }
    
    rt.mutex.Lock()
    defer rt.mutex.Unlock()
    
    bucket := rt.buckets[bucketIndex]
    
    // Check if already exists
    for i, c := range bucket {
        if c.ID == peerID {
            rt.buckets[bucketIndex][i] = contact
            return
        }
    }
    
    // Add new contact
    if len(bucket) < BucketSize {
        rt.buckets[bucketIndex] = append(bucket, contact)
    } else {
        // Replace oldest
        rt.buckets[bucketIndex][0] = contact
    }
}

func (rt *RoutingTable) FindClosest(target []byte, k int) []Contact {
    rt.mutex.RLock()
    defer rt.mutex.RUnlock()
    
    var contacts []Contact
    for _, bucket := range rt.buckets {
        contacts = append(contacts, bucket...)
    }
    
    // Sort by distance to target
    sort.Slice(contacts, func(i, j int) bool {
        di := XORDistance([]byte(contacts[i].ID), target)
        dj := XORDistance([]byte(contacts[j].ID), target)
        return compareDistance(di, dj) < 0
    })
    
    if k > len(contacts) {
        k = len(contacts)
    }
    return contacts[:k]
}

func (rt *RoutingTable) GetPeerCount() int {
    rt.mutex.RLock()
    defer rt.mutex.RUnlock()
    
    count := 0
    for _, bucket := range rt.buckets {
        count += len(bucket)
    }
    return count
}

// Helper functions
func XORDistance(a, b []byte) []byte {
    distance := make([]byte, KeySize)
    for i := 0; i < KeySize && i < len(a) && i < len(b); i++ {
        distance[i] = a[i] ^ b[i]
    }
    return distance
}

func getBucketIndex(distance []byte) int {
    for i, b := range distance {
        if b != 0 {
            for j := 7; j >= 0; j-- {
                if (b & (1 << j)) != 0 {
                    return i*8 + (7 - j)
                }
            }
        }
    }
    return KeySize*8 - 1
}

func compareDistance(a, b []byte) int {
    for i := 0; i < len(a) && i < len(b); i++ {
        if a[i] < b[i] {
            return -1
        } else if a[i] > b[i] {
            return 1
        }
    }
    return 0
}
