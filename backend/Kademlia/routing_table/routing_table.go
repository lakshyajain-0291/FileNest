package routing_table

import (
	"context"
	"fmt"
	"log"
	"math/bits"
	"math/rand"
	"net"
	"sort"
	"strconv"
	"sync"
	"time"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/multiformats/go-multiaddr"
)

const (
	// Kademlia constants
	BucketSize      = 20        // K value
	KeySize         = 64        // Updated for 64-bit integers
	AlphaValue      = 3         // Concurrency
	RefreshInterval = time.Hour // How often to refresh buckets
	PingTimeout     = 10 * time.Second
	uploadInterval  = 24 * time.Hour // how often to upload records to the database
)

// Contact represents a peer in the network
type Contact struct {
	ID         int // Changed from peer.ID to int
	Address    multiaddr.Multiaddr
	LastSeen   time.Time
	IsAlive    bool
	RTT        time.Duration // for calculating latency in communication
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
	localID int       // Changed from peer.ID to int
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

type CentralDB struct {
	gorm.Model
	peerID       int    `gorm:"primarykey"`
	multiaddress string `gorm:"not null"`
}

// a func for every peer to upload their peerid and multiaddr to the db
// this func will be called periodically as a goroutine inside the create routing table function
func (rt *RoutingTable) uploadtoCentralDB(db_name string, peerid int, host host.Host) {

	go func() {
        ticker := time.NewTicker(uploadInterval)
        defer ticker.Stop()
        
        db, err := gorm.Open(sqlite.Open(db_name), &gorm.Config{})
        if err != nil {
            log.Printf("Failed to connect to database: %v", err)
            return
        }
        
        for {
            select {
            case <-ticker.C:
                // Get CURRENT IP each time - not a parameter
                currentAddrs := host.Addrs()
                if len(currentAddrs) > 0 {
                    currentIP := currentAddrs[0].String()
					metadata := CentralDB{peerID: peerid, multiaddress: currentIP}
					results := db.Save(&metadata)

					if results.Error != nil{
						fmt.Printf("Could not upload the data to the central database: %v", results.Error)
					}else{
						fmt.Printf("Uploaded the data successfully")
					}
				
                }
                
            case <-rt.ctx.Done():
                return
            }
        }
    }()
}

// a func to fetch the routing table records and for a new peer
// this func will fetch the routing table only once when the peer is created initially.
// to ensure that you are not fetching from the central db once you have joined the network,
// just check if routing_table.db is empty or not.

func (rt *RoutingTable) fetchfromCentralDB(central_db_name string, local_db_name string) {
	// first read all the records
	central_db, err := gorm.Open(sqlite.Open(central_db_name), &gorm.Config{})
	if err != nil {
		log.Printf("Failed to connect to database: %v", err)
		return
	}

	routing_table_peers := rt.GetAllPeers()
	var peers []CentralDB
	result := central_db.Find(&peers)

	if result.Error != nil {
		log.Printf("Error reading records: %v", result.Error)
	} else {
		fmt.Printf("Found %d records\n", result.RowsAffected)
	}

	// then add these records to the routing table using add_peer
	if len(routing_table_peers) == 0{
		for _, peer := range peers{
		addr, err := multiaddr.NewMultiaddr(peer.multiaddress)
		if err != nil{
			fmt.Printf("Could not convert the string to multiaddr.Multiaddr: %v", err)
		}
		rt.AddPeer(peer.peerID, addr)
		}
	}else{
		rt.LoadFromDatabase()
	}
}

func NewRoutingTable(localID int, host host.Host) *RoutingTable {
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
			replacements: make([]*Contact, 0, BucketSize), // list of all the useless peers in the bucket
		}
	}

	// goroutine for updating the centralDB regularly
	go rt.uploadtoCentralDB("CentralDB.db", localID, host)

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
	rt.fetchfromCentralDB("CentralDB.db", "routing_table.db")

	// Start maintenance routines
	go rt.maintainBuckets(host)

	return rt
}

// XOR distance between two peer IDs (for 64-bit integers)
func calculateDistance(id1, id2 int) int {
	return id1 ^ id2
}

// getBucketIndex returns the bucket index for a given peer ID (64-bit version)
func (rt *RoutingTable) getBucketIndex(peerID int) int {
	distance := calculateDistance(rt.localID, peerID)
	if distance == 0 {
		return 0 // Same as local ID
	}

	// Use bits.LeadingZeros64 for efficient calculation
	leadingZeros := bits.LeadingZeros64(uint64(distance))
	return leadingZeros
}

// AddPeer adds or updates a peer in the routing table
func (rt *RoutingTable) AddPeer(peerID int, addr multiaddr.Multiaddr) error {
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
			PeerID:      fmt.Sprintf("%d", peerID), // Convert int to string
			Address:     addr.String(),
			LastSeen:    time.Now(),
		}
		rt.db.Where("peer_id = ?", fmt.Sprintf("%d", peerID)).
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

func (rt *RoutingTable) RemovePeer(peerID int) {
	bucketIndex := rt.getBucketIndex(peerID)
	bucket := rt.buckets[bucketIndex]

	bucket.mutex.Lock()
	defer bucket.mutex.Unlock()

	// Remove from database
	if rt.db != nil {
		rt.db.Where("peer_id = ?", fmt.Sprintf("%d", peerID)).Delete(&RoutingTableEntry{})
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

func (rt *RoutingTable) FindClosestPeers(targetID int, k int) []*Contact { // k closest peers
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
		return distI < distJ // Simple comparison for int
	})

	// Return at most k contacts
	if len(candidates) > k {
		candidates = candidates[:k]
	}

	return candidates
}

func (rt *RoutingTable) GetPeer(peerID int) (*Contact, bool) {
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
func PingPeer(ctx context.Context, contact *Contact, timeout time.Duration) error {
	// Extract host and port from multiaddr
	host, port, err := extractHostPort(contact.Address)
	if err != nil {
		return err
	}

	// Create connection with timeout
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%s", host, port), timeout)
	if err != nil {
		return fmt.Errorf("failed to connect to peer %d: %v", contact.ID, err)
	}
	defer conn.Close()

	// Update contact info on successful connection
	contact.IsAlive = true
	contact.LastSeen = time.Now()
	contact.RTT = time.Since(time.Now()) // You can measure actual RTT

	return nil
}

func extractHostPort(addr multiaddr.Multiaddr) (string, string, error) {
	host, err := addr.ValueForProtocol(multiaddr.P_IP4)
	if err != nil {
		return "", "", err
	}
	port, err := addr.ValueForProtocol(multiaddr.P_TCP)
	if err != nil {
		return "", "", err
	}
	return host, port, nil
}

// refreshBuckets refreshes stale buckets
func (rt *RoutingTable) refreshBuckets(host host.Host) {
	for i, bucket := range rt.buckets {
		bucket.mutex.RLock()
		timeSinceLast := time.Since(bucket.lastRefresh)
		bucket.mutex.RUnlock()

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

				// Use custom TCP ping instead of libp2p ping
				start := time.Now()
				err := PingPeer(ctx, c, 3*time.Second)
				if err == nil {
					// Measure RTT
					c.RTT = time.Since(start)
					c.IsAlive = true
					c.LastSeen = time.Now()

					mu.Lock()
					activeContacts = append(activeContacts, c)
					mu.Unlock()
				} else {
					log.Printf("Peer %d unreachable: %v", c.ID, err)
					c.IsAlive = false
					if rt.onPeerRemoved != nil {
						rt.onPeerRemoved(c)
					}
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
	// Generate a random target ID in the bucket's range for 64-bit
	var targetID int

	if bucketIndex == 0 {
		// For bucket 0, generate any random ID
		targetID = GenerateRandomID()
	} else {
		// Generate an ID that would fall into this specific bucket
		// by setting appropriate bit patterns
		distance := 1 << (63 - bucketIndex) // Create distance with MSB at correct position
		targetID = rt.localID ^ distance

		// Add some randomness to lower bits
		randomLower := rand.Intn(1 << (63 - bucketIndex))
		targetID ^= randomLower
	}

	log.Printf("Refreshing bucket %d with target ID %d", bucketIndex, targetID)

	// This would typically perform a FIND_NODE lookup to discover new peers
	// Implementation would go here
}

// GenerateRandomID generates a random 64-bit integer ID
func GenerateRandomID() int {
	return rand.Int() // This gives you a positive int in the full range
}

// RoutingTableEntry our database model
type RoutingTableEntry struct {
	gorm.Model
	BucketIndex int    `gorm:"index"`
	PeerID      string `gorm:"uniqueIndex"` // Keep as string for database
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
	rt.onPeerAdded = onAdded // callback functions for peer events
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
		peerID, err := strconv.Atoi(entry.PeerID) // Convert string to int
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
	fmt.Printf("Routing Table for peer %d:\n", rt.localID)

	totalPeers := 0
	for i, bucket := range rt.buckets {
		bucket.mutex.RLock()
		if len(bucket.contacts) > 0 {
			fmt.Printf("Bucket %d: %d contacts, %d replacements\n",
				i, len(bucket.contacts), len(bucket.replacements))
			for j, contact := range bucket.contacts {
				fmt.Printf("  [%d] %d (last seen: %s, alive: %t)\n",
					j, contact.ID,
					contact.LastSeen.Format("15:04:05"), contact.IsAlive)
			}
			totalPeers += len(bucket.contacts)
		}
		bucket.mutex.RUnlock()
	}

	fmt.Printf("Total peers: %d\n", totalPeers)
	fmt.Printf("Tag vectors: %d depths\n", len(rt.tagVectors))
	fmt.Printf("Stored embeddings: %d\n", len(rt.embeddings))
}
