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
    lastUpdated time.Time
}

func NewRoutingTable(nodeID []byte) *RoutingTable {
    return &RoutingTable{
        nodeID:      nodeID,
        buckets:     make([][]Contact, KeySize*8),
        lastUpdated: time.Now(),
    }
}


func (rt *RoutingTable) AddContact(peerID peer.ID, addrs []string) bool {
    if peerID == "" {
        return false
    }
    
    // Generate node ID for this peer (in production, this should come from the peer)
    peerNodeID := GenerateNodeID(peerID)
    distance := XORDistance(rt.nodeID, peerNodeID)
    bucketIndex := getBucketIndex(distance)
    
    contact := Contact{
        ID:       peerID,
        Addrs:    addrs,
        LastSeen: time.Now(),
        NodeID:   peerNodeID,
    }
    
    if !contact.IsValid() {
        return false
    }
    
    rt.mutex.Lock()
    defer rt.mutex.Unlock()
    
    bucket := rt.buckets[bucketIndex]
    
    // Check if already exists - update if so
    for i, c := range bucket {
        if c.ID == peerID {
            rt.buckets[bucketIndex][i] = contact
            rt.lastUpdated = time.Now()
            return true
        }
    }
    
    // Add new contact
    if len(bucket) < BucketSize {
        rt.buckets[bucketIndex] = append(bucket, contact)
        rt.lastUpdated = time.Now()
        return true
    }
    
    // Bucket full - LRU replacement
    oldestIndex := 0
    oldestTime := bucket[0].LastSeen
    for i, c := range bucket {
        if c.LastSeen.Before(oldestTime) {
            oldestIndex = i
            oldestTime = c.LastSeen
        }
    }
    
    rt.buckets[bucketIndex][oldestIndex] = contact
    rt.lastUpdated = time.Now()
    return true
}

func (rt *RoutingTable) RemoveContact(peerID peer.ID) bool {
    peerNodeID := GenerateNodeID(peerID)
    distance := XORDistance(rt.nodeID, peerNodeID)
    bucketIndex := getBucketIndex(distance)
    
    rt.mutex.Lock()
    defer rt.mutex.Unlock()
    
    bucket := rt.buckets[bucketIndex]
    for i, contact := range bucket {
        if contact.ID == peerID {
            rt.buckets[bucketIndex] = append(bucket[:i], bucket[i+1:]...)
            rt.lastUpdated = time.Now()
            return true
        }
    }
    return false
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
        di := XORDistance(contacts[i].NodeID, target)
        dj := XORDistance(contacts[j].NodeID, target)
        return compareDistance(di, dj) < 0
    })
    
    if k > len(contacts) {
        k = len(contacts)
    }
    return contacts[:k]
}

func (rt *RoutingTable) GetAllContacts() []Contact {
    rt.mutex.RLock()
    defer rt.mutex.RUnlock()
    
    var contacts []Contact
    for _, bucket := range rt.buckets {
        contacts = append(contacts, bucket...)
    }
    return contacts
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

func (rt *RoutingTable) GetBucketInfo() map[int]int {
    rt.mutex.RLock()
    defer rt.mutex.RUnlock()
    
    info := make(map[int]int)
    for i, bucket := range rt.buckets {
        if len(bucket) > 0 {
            info[i] = len(bucket)
        }
    }
    return info
}

// Cleanup removes stale contacts
func (rt *RoutingTable) Cleanup(maxAge time.Duration) int {
    rt.mutex.Lock()
    defer rt.mutex.Unlock()
    
    removed := 0
    now := time.Now()
    
    for i, bucket := range rt.buckets {
        var newBucket []Contact
        for _, contact := range bucket {
            if now.Sub(contact.LastSeen) <= maxAge {
                newBucket = append(newBucket, contact)
            } else {
                removed++
            }
        }
        rt.buckets[i] = newBucket
    }
    
    if removed > 0 {
        rt.lastUpdated = time.Now()
    }
    
    return removed
}

// Helper functions
func XORDistance(a, b []byte) []byte {
    distance := make([]byte, KeySize)
    minLen := len(a)
    if len(b) < minLen {
        minLen = len(b)
    }
    
    for i := 0; i < minLen; i++ {
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
