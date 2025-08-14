package routingtable

import (
	"bytes"
	"kademlia/pkg/helpers"
	"kademlia/pkg/types"
	"sort"
)

type RoutingTable struct {
	SelfNodeID []byte // Persistent unique node ID (used for XOR)
	SelfPeerID string // Ephemeral PeerID from libp2p
	Buckets    [][]types.PeerInfo
	K          int // Max bucket size
}

// NewRoutingTable initializes a new routing table
func NewRoutingTable(selfNodeID []byte, selfPeerID string, k int) *RoutingTable {
	numBuckets := len(selfNodeID) * 8
	buckets := make([][]types.PeerInfo, numBuckets)
	return &RoutingTable{
		SelfNodeID: selfNodeID,
		SelfPeerID: selfPeerID,
		Buckets:    buckets,
		K:          k,
	}
}

// Update adds or refreshes a peer in the correct bucket
func (rt *RoutingTable) Update(peer types.PeerInfo) {
	if bytes.Equal(rt.SelfNodeID, peer.NodeID) {
		return // don't add ourselves
	}

	bucketIndex := helpers.BucketIndex(rt.SelfNodeID, peer.NodeID)
	if bucketIndex < 0 || bucketIndex >= len(rt.Buckets) {
		return // invalid bucket index
	}

	bucket := rt.Buckets[bucketIndex]

	// If peer already exists, move to front
	for i, p := range bucket {
		if bytes.Equal(p.NodeID, peer.NodeID) {
			bucket = append([]types.PeerInfo{p}, append(bucket[:i], bucket[i+1:]...)...)
			rt.Buckets[bucketIndex] = bucket
			return
		}
	}

	// If not full, add to front
	if len(bucket) < rt.K {
		bucket = append([]types.PeerInfo{peer}, bucket...)
		
		rt.Buckets[bucketIndex] = bucket
		return
	} else {
		// If full, remove last and insert at front
		bucket = append([]types.PeerInfo{peer}, bucket[:rt.K-1]...)
		rt.Buckets[bucketIndex] = bucket
	}

}

// FindClosest returns count closest peers to target NodeID
func (rt *RoutingTable) FindClosest(targetID []byte, count int) []types.PeerInfo {
	var allPeers []types.PeerInfo
	for _, bucket := range rt.Buckets {
		allPeers = append(allPeers, bucket...)
	}

	// Sort by XOR distance of NodeIDs
	sort.Slice(allPeers, func(i, j int) bool {
		distI := helpers.XORDistance(targetID, allPeers[i].NodeID)
		distJ := helpers.XORDistance(targetID, allPeers[j].NodeID)
		return distI.Cmp(distJ) < 0
	})

	if count > len(allPeers) {
		count = len(allPeers)
	}

	return allPeers[:count]
}

// Remove deletes a peer from its bucket
func (rt *RoutingTable) Remove(nodeID []byte) {
	bucketIndex := helpers.BucketIndex(rt.SelfNodeID, nodeID)
	if bucketIndex < 0 || bucketIndex >= len(rt.Buckets) {
		return
	}

	bucket := rt.Buckets[bucketIndex]
	for i, p := range bucket {
		if bytes.Equal(p.NodeID, nodeID) {
			bucket = append(bucket[:i], bucket[i+1:]...)
			rt.Buckets[bucketIndex] = bucket
			return
		}
	}
}
