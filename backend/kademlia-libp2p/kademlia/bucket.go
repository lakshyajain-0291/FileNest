package kademlia

import (
    "sort"
    "sync"
    "time"
    "github.com/libp2p/go-libp2p/core/peer"
)

const (
    BucketSize = 20 // K value in Kademlia
)

// Contact represents a peer contact
type Contact struct {
    ID       peer.ID
    Addrs    []string
    LastSeen time.Time
    Distance []byte
}

// Bucket represents a k-bucket in the routing table
type Bucket struct {
    mutex    sync.RWMutex
    contacts []Contact
    lastChanged time.Time
}

// NewBucket creates a new k-bucket
func NewBucket() *Bucket {
    return &Bucket{
        contacts: make([]Contact, 0, BucketSize),
        lastChanged: time.Now(),
    }
}

// AddContact adds or updates a contact in the bucket
func (b *Bucket) AddContact(contact Contact) bool {
    b.mutex.Lock()
    defer b.mutex.Unlock()

    // Check if contact already exists
    for i, c := range b.contacts {
        if c.ID == contact.ID {
            // Update existing contact
            b.contacts[i] = contact
            b.lastChanged = time.Now()
            return true
        }
    }

    // If bucket is not full, add the contact
    if len(b.contacts) < BucketSize {
        b.contacts = append(b.contacts, contact)
        b.lastChanged = time.Now()
        return true
    }

    // Bucket is full - implement LRU eviction
    // Find the least recently seen contact
    oldestIndex := 0
    oldestTime := b.contacts[0].LastSeen
    for i, c := range b.contacts {
        if c.LastSeen.Before(oldestTime) {
            oldestIndex = i
            oldestTime = c.LastSeen
        }
    }

    // Replace oldest contact
    b.contacts[oldestIndex] = contact
    b.lastChanged = time.Now()
    return true
}

// RemoveContact removes a contact from the bucket
func (b *Bucket) RemoveContact(peerID peer.ID) bool {
    b.mutex.Lock()
    defer b.mutex.Unlock()

    for i, contact := range b.contacts {
        if contact.ID == peerID {
            b.contacts = append(b.contacts[:i], b.contacts[i+1:]...)
            b.lastChanged = time.Now()
            return true
        }
    }
    return false
}

// GetContacts returns all contacts in the bucket
func (b *Bucket) GetContacts() []Contact {
    b.mutex.RLock()
    defer b.mutex.RUnlock()

    contacts := make([]Contact, len(b.contacts))
    copy(contacts, b.contacts)
    return contacts
}

// Size returns the number of contacts in the bucket
func (b *Bucket) Size() int {
    b.mutex.RLock()
    defer b.mutex.RUnlock()
    return len(b.contacts)
}

// GetClosestContacts returns the k closest contacts to a given distance
func (b *Bucket) GetClosestContacts(target []byte, k int) []Contact {
    b.mutex.RLock()
    defer b.mutex.RUnlock()

    if k > len(b.contacts) {
        k = len(b.contacts)
    }

    contacts := make([]Contact, len(b.contacts))
    copy(contacts, b.contacts)

    // Sort by XOR distance to target
    sort.Slice(contacts, func(i, j int) bool {
        distI := XOR(contacts[i].Distance, target)
        distJ := XOR(contacts[j].Distance, target)
        return CompareBytes(distI, distJ) < 0
    })

    if k > len(contacts) {
        k = len(contacts)
    }

    return contacts[:k]
}
