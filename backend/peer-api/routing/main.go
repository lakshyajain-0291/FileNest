package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"

	"github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multihash"
)

type Node struct {
	host   host.Host
	dht    *dht.IpfsDHT
	rt     *RoutingTable
	ctx    context.Context
	cancel context.CancelFunc
}

func newNOde(port int) (*Node, error) {
	ctx, cancel := context.WithCancel(context.Background())

	//New constructs a new libp2p node with the given options, falling back on reasonable defaults.
	h, err := libp2p.New(libp2p.ListenAddrStrings(fmt.Sprintf("/ip4/127.0.0.1/tcp/%d", port))) //listen on unparsed strings

	if err != nil {
		cancel()
		return nil, err
	}

	// Create DHT
	kadDHT, err := dht.New(ctx, h)
	if err != nil {
		h.Close()
		cancel()
		return nil, err
	}

}

type KBucket struct {
	peers []peer.ID
	k     int // bucket size. there are 160 buckets of which each can hold k peers (from each distant bucket)
}

// new k-bucket with given capacity
func NewKBucket(k int) *KBucket {
	return &KBucket{

		peers: make([]peer.ID, 0, k),
		k:     k,
	}
}

// adds a peer to the bucket if there's space
func (kb *KBucket) Add(p peer.ID) bool {
	// Check if peer already exists
	for _, existing := range kb.peers {
		if existing == p {
			return false
		}
	}

	// if there's space
	if len(kb.peers) < kb.k {
		kb.peers = append(kb.peers, p)
		return true
	}

	return false
}

// remove removes a peer from the bucket
func (kb *KBucket) Remove(p peer.ID) {
	for i, existing := range kb.peers {
		if existing == p {
			kb.peers = append(kb.peers[:i], kb.peers[i+1:]...) //skip the ith peer
			return
		}
	}
}

// GetPeers returns all peers in the bucket
func (kb *KBucket) GetPeers() []peer.ID {
	return kb.peers
}

type RoutingTable struct { // represents a Kademlia routing table
	localID peer.ID    //owner of the routing table
	buckets []*KBucket //160 buckets
	k       int        // bucket size
}

func NewRoutingTable(localID peer.ID, k int) *RoutingTable { // new routing table

	buckets := make([]*KBucket, 160) //slice of 160 buckets
	for i := range buckets {
		buckets[i] = NewKBucket(k)
	}

	return &RoutingTable{
		localID: localID,
		buckets: buckets,
		k:       k,
	}
}

// XOR distance between two peer IDs
func xorDistance(a, b peer.ID) []byte {
	aBytes := []byte(a)
	bBytes := []byte(b)

	maxLen := len(aBytes)
	if len(bBytes) > maxLen {
		maxLen = len(bBytes)
	}

	result := make([]byte, maxLen) //whichevers is longer
	for i := 0; i < maxLen; i++ {
		var aByte, bByte byte
		if i < len(aBytes) {
			aByte = aBytes[i]
		}
		if i < len(bBytes) {
			bByte = bBytes[i]
		}
		result[i] = aByte ^ bByte
	}

	return result
}

func (rt *RoutingTable) getBucketIndex(peerID peer.ID) int { // returns the bucket index for a peer based on XOR distance
	distance := xorDistance(rt.localID, peerID)

	// Find the position of the first bit that differs (counting from left)
	for i, b := range distance {
		if b != 0 { //bit differs
			// Find the position of the leftmost 1 bit
			for j := 7; j >= 0; j-- {
				if (b & (1 << j)) != 0 {
					return i*8 + (7 - j) //that leftmost bit position is the bucket index
				}
			}
		}
	}

	return len(distance)*8 - 1 // If all bits are the same, return the last bucket
}

// adds a peer to the routing table
func (rt *RoutingTable) AddPeer(peerID peer.ID) bool {
	if peerID == rt.localID {
		return false // dont add ourselves
	}

	bucketIndex := rt.getBucketIndex(peerID) //see where to put the peer
	return rt.buckets[bucketIndex].Add(peerID)
}

// RemovePeer removes a peer from the routing table
func (rt *RoutingTable) RemovePeer(peerID peer.ID) {
	bucketIndex := rt.getBucketIndex(peerID)
	rt.buckets[bucketIndex].Remove(peerID)
}

type PeerDistance struct { //reepresents a peer with its distance
	Peer     peer.ID
	Distance []byte
}

func (rt *RoutingTable) GetClosestPeers(target peer.ID, count int) []peer.ID { //  k closest peers to a given peer ID
	var candidates []PeerDistance

	for _, bucket := range rt.buckets { // Collect all peers from all buckets
		for _, p := range bucket.GetPeers() {
			candidates = append(candidates, PeerDistance{
				Peer:     p,
				Distance: xorDistance(target, p),
			})
		}
	}

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return compareDistance(candidates[i].Distance, candidates[j].Distance) < 0
	})

	result := make([]peer.ID, 0, count) // Return count closest peers
	for i := 0; i < len(candidates) && i < count; i++ {
		result = append(result, candidates[i].Peer)
	}

	return result
}

// compareDistance compares two distance bytearrays
// retun -1 if a < b, 0 if a == b, 1 if a > b
func compareDistance(a, b []byte) int {
	maxLen := len(a)
	if len(b) > maxLen {
		maxLen = len(b)
	}

	for i := 0; i < maxLen; i++ {
		var aByte, bByte byte
		if i < len(a) {
			aByte = a[i]
		}
		if i < len(b) {
			bByte = b[i]
		}

		if aByte < bByte {
			return -1
		} else if aByte > bByte {
			return 1
		}
	}

	return 0
}

// GetAllPeers returns all peers in the routing table
func (rt *RoutingTable) GetAllPeers() []peer.ID {
	var allPeers []peer.ID

	for _, bucket := range rt.buckets {
		allPeers = append(allPeers, bucket.GetPeers()...)
	}
	return allPeers
}

// createPeerID creates a peer ID from a stri
func createPeerID(s string) peer.ID {
	h := sha256.Sum256([]byte(s))
	mh, _ := multihash.Encode(h[:], multihash.SHA2_256)
	return peer.ID(mh)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("go run main.go <port> [bootstrap_peer_multiaddr]")
		fmt.Println("Example: go run main.go 8001")
		fmt.Println("Example: go run main.go 8002 /ip4/127.0.0.1/tcp/8001/p2p/QmBootstrapPeerID")
		os.Exit(1)
	}

	port, err := strconv.Atoi(os.Args[1])
	if err != nil {
		log.Fatal("invalid port", err)

	}
	node, err := newNOde(port)
	if err != nil {
		log.Fatal("Node couldnt be created")

	}
	defer node.Close()

}
