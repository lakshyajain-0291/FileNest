package routing_table

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log"
	"math/big"
	"sort"
	"sync"
	"time"
<<<<<<< HEAD

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"

=======
	"github.com/libp2p/go-libp2p/p2p/protocol/ping"
	"github.com/libp2p/go-libp2p/core/host"
>>>>>>> c1ac24136fe57d94ff4a7fca16aa4e44cc99726d
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
)

const (
	// Kademlia constants
	BucketSize      = 20 // K value
	KeySize         = 256
	AlphaValue      = 3         // Concurrency
	RefreshInterval = time.Hour // How often to refresh buckets
	PingTimeout     = 10 * time.Second
)

// Contact represents a peer in the network
type Contact struct {
	ID         peer.ID
	Address    multiaddr.Multiaddr
	LastSeen   time.Time
	IsAlive    bool
	RTT        time.Duration //for caclulating latency in communication
	Depth      int
	TagVectors []TagVector // For FileNest tagging system
}

// tagging vector at different depths
type TagVector struct {
	ID        string
	Vector    []float64
	Count     int
	Centroid  []float64
	Depth     int
	Threshold float64
}

// Bucket represents a k-bucket in the routing table
type Bucket struct {
	contacts     []*Contact   // List of peers in the bucket
	mutex        sync.RWMutex // mutex for concurrent access
	lastRefresh  time.Time
	replacements []*Contact // list of replacement peers
}

// to implement the main routing
type RoutingTable struct {
	localID peer.ID   // Local peer
	buckets []*Bucket // Kademlia buckets
	mutex   sync.RWMutex
	ctx     context.Context
	cancel  context.CancelFunc

	tagVectors map[int][]TagVector  // Depth mapped to list of tag vectors
	embeddings map[string][]float64 // File hash mapped to embedding

	// Database connection
	db *gorm.DB

	// Callbacks
	onPeerAdded   func(contact *Contact)
	onPeerRemoved func(contact *Contact)
}

func NewRoutingTable(localID peer.ID, host host.Host) *RoutingTable {
	ctx, cancel := context.WithCancel(context.Background())

	rt := &RoutingTable{
		localID:    localID,
		buckets:    make([]*Bucket, KeySize),
		ctx:        ctx,
		cancel:     cancel,
		tagVectors: make(map[int][]TagVector),
		embeddings: make(map[string][]float64),
	}

	// Initialize buckets
	for i := 0; i < KeySize; i++ {
		rt.buckets[i] = &Bucket{
			contacts:     make([]*Contact, 0, BucketSize),
			replacements: make([]*Contact, 0, BucketSize), //list of all the useless peers in the bucet
		}
	}

	// Initialize database connection
	db, err := gorm.Open(sqlite.Open("routing_table.db"), &gorm.Config{})
	if err != nil {
		log.Printf("Failed to connect to database: %v", err)
	} else {
		rt.db = db
		// Auto migrate the schema
		err = db.AutoMigrate(&RoutingTableEntry{})
		if err != nil {
			log.Printf("Failed to migrate database: %v", err)
		}
	}

	// Load existing entries from database
	//rt.LoadFromDatabase()

	// Start maintenance routines
	go rt.maintainBuckets(host)

	return rt
}

// XOR distance between two peer IDs
func calculateDistance(id1, id2 peer.ID) *big.Int {
	hash1 := sha256.Sum256([]byte(id1)) //first hash the peer ids
	hash2 := sha256.Sum256([]byte(id2))
	// dist := subtle.XOR(bytes.NewSlice(hash1[:]), bytes.NewSlice(hash2[:])) XOR function not being recognised

	dist := new(big.Int)
	for i := 0; i < 32; i++ { //iterate over each of the 32 bytes of hash value
		dist.SetBit(dist, i*8+7, uint(hash1[i]^hash2[i])&1)    //i*8+7 is the index of the bit in dist.&1: This is performing a bitwise AND operation between the result of the XOR operation and the value 1. This is essentially extracting the least significant bit (LSB) of the result.
		dist.SetBit(dist, i*8+6, uint(hash1[i]^hash2[i])>>1&1) //The >>1 operation shifts the bits one position to the right, effectively dividing the result by 2.
		dist.SetBit(dist, i*8+5, uint(hash1[i]^hash2[i])>>2&1)
		dist.SetBit(dist, i*8+4, uint(hash1[i]^hash2[i])>>3&1)
		dist.SetBit(dist, i*8+3, uint(hash1[i]^hash2[i])>>4&1)
		dist.SetBit(dist, i*8+2, uint(hash1[i]^hash2[i])>>5&1)
		dist.SetBit(dist, i*8+1, uint(hash1[i]^hash2[i])>>6&1)
		dist.SetBit(dist, i*8+0, uint(hash1[i]^hash2[i])>>7&1)
	}
	return dist
}

// getBucketIndex returns the bucket index for a given peer ID
func (rt *RoutingTable) getBucketIndex(peerID peer.ID) int {
	distance := calculateDistance(rt.localID, peerID)
	if distance.Sign() == 0 {
		return 0 // Same as local ID
	}

	// Find the position of the most significant bit
	bitLen := distance.BitLen()
	if bitLen == 0 {
		return 0
	}
	return KeySize - bitLen
}

// AddPeer adds or updates a peer in the routing table
func (rt *RoutingTable) AddPeer(peerID peer.ID, addr multiaddr.Multiaddr) error {
	if peerID == rt.localID {
		return nil // Don't add ourselves
	}

	bucketIndex := rt.getBucketIndex(peerID)
	bucket := rt.buckets[bucketIndex]

	bucket.mutex.Lock()
	defer bucket.mutex.Unlock()

	// Update database
	if rt.db != nil {
		entry := RoutingTableEntry{
			BucketIndex: bucketIndex,
			PeerID:      peerID.String(),
			Address:     addr.String(),
			LastSeen:    time.Now(),
		}
		rt.db.Where("peer_id = ?", peerID.String()).
			Assign(entry).
			FirstOrCreate(&entry)
	}

	// Check if peer already exists
	for i, contact := range bucket.contacts {
		if contact.ID == peerID {
			// Update existing contact
			contact.Address = addr
			contact.LastSeen = time.Now()
			contact.IsAlive = true

			// Move to front like a queue. last seen contact is at the top
			bucket.contacts = append([]*Contact{contact},
				append(bucket.contacts[:i], bucket.contacts[i+1:]...)...)
			return nil
		}
	}

	// Create new contact
	contact := &Contact{
		ID:       peerID,
		Address:  addr,
		LastSeen: time.Now(),
		IsAlive:  true,
	}

	// If bucket is not full, add contact
	if len(bucket.contacts) < BucketSize {
		bucket.contacts = append([]*Contact{contact}, bucket.contacts...)
		if rt.onPeerAdded != nil {
			rt.onPeerAdded(contact)
		}
		return nil
	}

	// Bucket is full, try to ping the least recently seen contact
	leastRecent := bucket.contacts[len(bucket.contacts)-1]
	if time.Since(leastRecent.LastSeen) > PingTimeout {
		// Replace least recent with new contact
		bucket.contacts[len(bucket.contacts)-1] = contact
		if rt.onPeerRemoved != nil {
			rt.onPeerRemoved(leastRecent)
		}
		if rt.onPeerAdded != nil {
			rt.onPeerAdded(contact)
		}
	} else {
		// Add to replacement cache
		if len(bucket.replacements) < BucketSize {
			bucket.replacements = append(bucket.replacements, contact)
		}
	}

	return nil
}

func (rt *RoutingTable) RemovePeer(peerID peer.ID) {
	bucketIndex := rt.getBucketIndex(peerID)
	bucket := rt.buckets[bucketIndex]

	bucket.mutex.Lock()
	defer bucket.mutex.Unlock()

	// Remove from database
	if rt.db != nil {
		rt.db.Where("peer_id = ?", peerID.String()).Delete(&RoutingTableEntry{})
	}

	for i, contact := range bucket.contacts {
		if contact.ID == peerID {
			// remove contact
			bucket.contacts = append(bucket.contacts[:i], bucket.contacts[i+1:]...)

			// Try to add a replacement - from the cache
			if len(bucket.replacements) > 0 {
				replacement := bucket.replacements[0]
				bucket.replacements = bucket.replacements[1:]
				bucket.contacts = append(bucket.contacts, replacement)
				if rt.onPeerAdded != nil {
					rt.onPeerAdded(replacement)
				}
			}

			if rt.onPeerRemoved != nil {
				rt.onPeerRemoved(contact)
			}
			return
		}
	}
}

func (rt *RoutingTable) FindClosestPeers(targetID peer.ID, k int) []*Contact { //k closest peers
	bucketIndex := rt.getBucketIndex(targetID)

	var candidates []*Contact

	// start with the appropriate bucket
	rt.buckets[bucketIndex].mutex.RLock()
	for _, contact := range rt.buckets[bucketIndex].contacts {
		if contact.IsAlive {
			candidates = append(candidates, contact)
		}
	}
	rt.buckets[bucketIndex].mutex.RUnlock()

	// Expand to neighboring buckets if needed
	for i := 1; len(candidates) < k && (bucketIndex-i >= 0 || bucketIndex+i < KeySize); i++ {
		if bucketIndex-i >= 0 {
			rt.buckets[bucketIndex-i].mutex.RLock()
			for _, contact := range rt.buckets[bucketIndex-i].contacts {
				if contact.IsAlive {
					candidates = append(candidates, contact)
				}
			}
			rt.buckets[bucketIndex-i].mutex.RUnlock()
		}

		if bucketIndex+i < KeySize {
			rt.buckets[bucketIndex+i].mutex.RLock()
			for _, contact := range rt.buckets[bucketIndex+i].contacts {
				if contact.IsAlive {
					candidates = append(candidates, contact)
				}
			}
			rt.buckets[bucketIndex+i].mutex.RUnlock()
		}
	}

	// Sort by distance to target
	sort.Slice(candidates, func(i, j int) bool {
		distI := calculateDistance(targetID, candidates[i].ID)
		distJ := calculateDistance(targetID, candidates[j].ID)
		return distI.Cmp(distJ) < 0
	})

	// Return at most k contacts
	if len(candidates) > k {
		candidates = candidates[:k]
	}

	return candidates
}

func (rt *RoutingTable) GetPeer(peerID peer.ID) (*Contact, bool) {
	bucketIndex := rt.getBucketIndex(peerID)
	bucket := rt.buckets[bucketIndex]

	bucket.mutex.RLock()
	defer bucket.mutex.RUnlock()

	for _, contact := range bucket.contacts {
		if contact.ID == peerID {
			return contact, true
		}
	}

	return nil, false
}

func (rt *RoutingTable) GetAllPeers() []*Contact {
	var allPeers []*Contact

	for _, bucket := range rt.buckets {
		bucket.mutex.RLock()
		for _, contact := range bucket.contacts {
			if contact.IsAlive {
				allPeers = append(allPeers, contact)
			}
		}
		bucket.mutex.RUnlock()
	}

	return allPeers
}

// maintainBuckets performs periodic maintenance on buckets
func (rt *RoutingTable) maintainBuckets(host host.Host) {
	ticker := time.NewTicker(RefreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rt.ctx.Done():
			return
		case <-ticker.C:
			rt.refreshBuckets(host)
		}
	}
}

// acts as the ping rpc
func Ping(ctx context.Context, host host.Host, contact *Contact) (ping.Result, error) {
	fullAddr := multiaddr.Join(contact.Address, multiaddr.StringCast("/p2p/"+contact.ID.String()))
	addrInfo, err := peer.AddrInfoFromP2pAddr(fullAddr)
	if err != nil {
		return ping.Result{}, fmt.Errorf("invalid addrinfo: %w", err)
	}

	if err := host.Connect(ctx, *addrInfo); err != nil {
		return ping.Result{}, fmt.Errorf("connect failed: %w", err)
	}

	pingService := ping.NewPingService(host)
	result := <-pingService.Ping(ctx, addrInfo.ID)

	if result.Error != nil {
		return result, result.Error
	}

	return result, nil
}

// refreshBuckets refreshes stale buckets

<<<<<<< HEAD
		// Check if bucket needs refresh
		if time.Since(bucket.lastRefresh) > RefreshInterval {

			go rt.performBucketRefresh(i)
			bucket.lastRefresh = time.Now()
		}
=======
func (rt *RoutingTable) refreshBuckets(host host.Host) {
    for i, bucket := range rt.buckets {
        bucket.mutex.RLock()
        timeSinceLast := time.Since(bucket.lastRefresh)
        bucket.mutex.RUnlock()
>>>>>>> c1ac24136fe57d94ff4a7fca16aa4e44cc99726d

        if timeSinceLast > RefreshInterval {
            go rt.performBucketRefresh(i)
            bucket.mutex.Lock()
            bucket.lastRefresh = time.Now()
            bucket.mutex.Unlock()
        }

        // Ping peers concurrently and prune unresponsive ones
        bucket.mutex.RLock()
        contacts := make([]*Contact, len(bucket.contacts))
        copy(contacts, bucket.contacts)
        bucket.mutex.RUnlock()

        var wg sync.WaitGroup
        var mu sync.Mutex
        var activeContacts []*Contact

        ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
        defer cancel()

        for _, contact := range contacts {
            wg.Add(1)
            go func(c *Contact) {
                defer wg.Done()
                result, err := Ping(ctx, host, c)
                if err == nil && result.Error == nil {
                    mu.Lock()
                    activeContacts = append(activeContacts, c)
                    mu.Unlock()
                } else if rt.onPeerRemoved != nil {
                    rt.onPeerRemoved(c)
                }
            }(contact)
        }
        wg.Wait()

        bucket.mutex.Lock()
        bucket.contacts = activeContacts
        bucket.mutex.Unlock()
    }
}
// performBucketRefresh performs a lookup to refresh a bucket
func (rt *RoutingTable) performBucketRefresh(bucketIndex int) {
	// This would typically generate a random target ID in the bucket's range
	// and perform a FIND_NODE lookup to discover new peers

	log.Printf("Refreshing bucket %d", bucketIndex)
}

// RoutingTableEntry our databse model
type RoutingTableEntry struct {
	gorm.Model
	BucketIndex int    `gorm:"index"`
	PeerID      string `gorm:"uniqueIndex"`
	Address     string
	LastSeen    time.Time
}

// FileNest specific methods

// AddTagVector adds a tagging vector
func (rt *RoutingTable) AddTagVector(depth int, vector TagVector) {
	rt.mutex.Lock()
	defer rt.mutex.Unlock()

	if rt.tagVectors[depth] == nil {
		rt.tagVectors[depth] = make([]TagVector, 0)
	}

	rt.tagVectors[depth] = append(rt.tagVectors[depth], vector)
}

// FindClosestTagVector finds the most similar tag vector at a given depth
func (rt *RoutingTable) FindClosestTagVector(depth int, queryVector []float64, threshold float64) (*TagVector, float64) {
	rt.mutex.RLock()
	defer rt.mutex.RUnlock()

	vectors, exists := rt.tagVectors[depth]
	if !exists {
		return nil, 0.0
	}

	var bestVector *TagVector
	var bestSimilarity float64 = -1.0

	for i := range vectors {
		similarity := cosineSimilarity(queryVector, vectors[i].Vector)
		if similarity > bestSimilarity && similarity >= threshold {
			bestSimilarity = similarity
			bestVector = &vectors[i]
		}
	}

	return bestVector, bestSimilarity
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}

	return dotProduct / (normA * normB)
}

func (rt *RoutingTable) StoreEmbedding(fileHash string, embedding []float64) {
	rt.mutex.Lock()
	defer rt.mutex.Unlock()

	rt.embeddings[fileHash] = embedding
}

func (rt *RoutingTable) GetEmbedding(fileHash string) ([]float64, bool) {
	rt.mutex.RLock()
	defer rt.mutex.RUnlock()

	embedding, exists := rt.embeddings[fileHash]
	return embedding, exists
}

func (rt *RoutingTable) SetCallbacks(onAdded, onRemoved func(*Contact)) {
	rt.onPeerAdded = onAdded //callback functions for peer events
	rt.onPeerRemoved = onRemoved
}

func (rt *RoutingTable) Size() int {
	count := 0
	for _, bucket := range rt.buckets {
		bucket.mutex.RLock()
		count += len(bucket.contacts)
		bucket.mutex.RUnlock()
	}
	return count
}

// LoadFromDatabase loads the routing table entries from SQLite database
func (rt *RoutingTable) LoadFromDatabase() error {
	db, err := gorm.Open(sqlite.Open("routing_table.db"), &gorm.Config{})
	if err != nil {
		return fmt.Errorf("failed to open database: %v", err)
	}

	var entries []RoutingTableEntry
	result := db.Find(&entries)
	if result.Error != nil {
		return fmt.Errorf("failed to load entries: %v", result.Error)
	}

	for _, entry := range entries {
		peerID, err := peer.Decode(entry.PeerID)
		if err != nil {
			continue
		}
		addr, err := multiaddr.NewMultiaddr(entry.Address)
		if err != nil {
			continue
		}

		rt.AddPeer(peerID, addr)
	}

	return nil
}

// Close stops the routing table and cleanup
func (rt *RoutingTable) Close() {
	rt.cancel()
}

func (rt *RoutingTable) GetBucketInfo(index int) (int, int, time.Time) {
	if index < 0 || index >= KeySize {
		return 0, 0, time.Time{}
	}

	bucket := rt.buckets[index]
	bucket.mutex.RLock()
	defer bucket.mutex.RUnlock()

	return len(bucket.contacts), len(bucket.replacements), bucket.lastRefresh
}

// prints the current state of the routing table
func (rt *RoutingTable) PrintRoutingTable() {
	fmt.Printf("routing Table for peer %s:\n", rt.localID[:8])

	totalPeers := 0
	for i, bucket := range rt.buckets {
		bucket.mutex.RLock()
		if len(bucket.contacts) > 0 {
			fmt.Printf("Bucket %d: %d contacts, %d replacements\n",
				i, len(bucket.contacts), len(bucket.replacements))
			for j, contact := range bucket.contacts {
				fmt.Printf("  [%d] %s (last seen: %s, alive: %t)\n",
					j, contact.ID[:8],
					contact.LastSeen.Format("15:04:05"), contact.IsAlive)
			}
			totalPeers += len(bucket.contacts)
		}
		bucket.mutex.RUnlock()
	}

	fmt.Printf("Total peers: %d\n", totalPeers)
	fmt.Printf("Tag vectors: %d depths\n", len(rt.tagVectors))
	fmt.Printf("stord embeddings: %d\n", len(rt.embeddings))

}
