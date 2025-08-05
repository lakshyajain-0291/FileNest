package kademlia

import (
	"sort"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

const (
	KeySize = 32 // 256-bit keys (SHA-256)
	Alpha   = 3  // Concurrency parameter
)

// RoutingTable represents the Kademlia routing table
type RoutingTable struct {
	mutex   sync.RWMutex
	localID []byte
	buckets []*Bucket // 256 buckets for 256-bit keys
}

// NewRoutingTable creates a new routing table
func NewRoutingTable(localID []byte) *RoutingTable {
	buckets := make([]*Bucket, KeySize*8)
	for i := range buckets {
		buckets[i] = NewBucket()
	}

	return &RoutingTable{
		localID: localID,
		buckets: buckets,
	}
}

// AddContact adds a peer to the appropriate bucket
func (rt *RoutingTable) AddContact(peerID peer.ID, addrs []string) {
	peerIDBytes := []byte(peerID)
	distance := XOR(rt.localID, peerIDBytes)
	bucketIndex := GetBucketIndex(distance)

	contact := Contact{
		ID:       peerID,
		Addrs:    addrs,
		Distance: distance,
		LastSeen: time.Now(),
	}

	rt.mutex.Lock()
	defer rt.mutex.Unlock()
	rt.buckets[bucketIndex].AddContact(contact)
}

// RemoveContact removes a peer from the routing table
func (rt *RoutingTable) RemoveContact(peerID peer.ID) {
	peerIDBytes := []byte(peerID)
	distance := XOR(rt.localID, peerIDBytes)
	bucketIndex := GetBucketIndex(distance)

	rt.mutex.Lock()
	defer rt.mutex.Unlock()
	rt.buckets[bucketIndex].RemoveContact(peerID)
}

// FindClosestContacts finds the k closest contacts to a target key
func (rt *RoutingTable) FindClosestContacts(target []byte, k int) []Contact {
	rt.mutex.RLock()
	defer rt.mutex.RUnlock()

	distance := XOR(rt.localID, target)
	bucketIndex := GetBucketIndex(distance)

	var allContacts []Contact

	// Start with the appropriate bucket
	contacts := rt.buckets[bucketIndex].GetContacts()
	allContacts = append(allContacts, contacts...)

	// Expand to neighboring buckets if needed
	for i := 1; len(allContacts) < k && i < len(rt.buckets); i++ {
		// Check bucket to the left
		leftIndex := bucketIndex - i
		if leftIndex >= 0 {
			contacts := rt.buckets[leftIndex].GetContacts()
			allContacts = append(allContacts, contacts...)
		}

		// Check bucket to the right
		rightIndex := bucketIndex + i
		if rightIndex < len(rt.buckets) {
			contacts := rt.buckets[rightIndex].GetContacts()
			allContacts = append(allContacts, contacts...)
		}
	}

	// Sort by distance and return k closest
	return GetKClosest(allContacts, target, k)
}

// GetBucketIndex calculates the bucket index for a distance
func GetBucketIndex(distance []byte) int {
	for i, b := range distance {
		if b != 0 {
			// Find the position of the most significant bit
			for j := 7; j >= 0; j-- {
				if (b & (1 << j)) != 0 {
					return i*8 + (7 - j)
				}
			}
		}
	}
	return KeySize*8 - 1 // All bits are zero
}

// Helper functions for distance calculations
func XOR(a, b []byte) []byte {
	result := make([]byte, len(a))
	for i := 0; i < len(a) && i < len(b); i++ {
		result[i] = a[i] ^ b[i]
	}
	return result
}

func CompareBytes(a, b []byte) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] < b[i] {
			return -1
		} else if a[i] > b[i] {
			return 1
		}
	}
	return 0
}

func GetKClosest(contacts []Contact, target []byte, k int) []Contact {
	// Sort contacts by XOR distance to target
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
