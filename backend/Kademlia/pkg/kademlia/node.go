package kademlia

import (
	"crypto/sha1"
	"encoding/binary"
	"final/backend/pkg/helpers"
	"final/backend/pkg/storage"
	"final/backend/pkg/types"
	"fmt"
	"log"
	"math"
	"sort"
	"time"
)

type KademliaNode struct {
	// RoutingTable returns the node's routing table (exported getter)
	NodeID       []byte // Persistent NodeID
	PeerID       string // Ephemeral libp2p PeerID
	routingTable *RoutingTable // stores nodeIDs which have contacted the Node before
	storage      storage.Interface // stores nodeIDs which match the TV of this node
	network      NetworkInterface
}

func NewKademliaNode(nodeID []byte, peerID string, network NetworkInterface, dbPath string) (*KademliaNode, error) {
	// RoutingTable returns the node's routing table (exported getter)
	log.Printf("NewKademliaNode: initializing node with peerID=%s", peerID)

	// ADD THIS VALIDATION:
	if len(nodeID) != 20 {
		err := fmt.Errorf("nodeID must be 20 bytes (160 bits), got %d", len(nodeID))
		log.Printf("NewKademliaNode: invalid nodeID length: %v", err)
		return nil, err
	}

	sqliteStorage, err := storage.NewSQLiteStorage(dbPath)
	if err != nil {
		log.Printf("NewKademliaNode: failed to initialize sqlite storage: %v", err)
		return nil, err
	}
	log.Printf("NewKademliaNode: sqlite storage initialized at %s", dbPath)

	node := &KademliaNode{
		NodeID:       nodeID,
		PeerID:       peerID,
		routingTable: NewRoutingTable(nodeID, peerID, 20), // K=20
		storage:      sqliteStorage,                       // Initialize storage
		network:      network,
	}

	log.Printf("NewKademliaNode: node created successfully: %x (peerID=%s)", nodeID[:8], peerID)
	return node, nil
}

func (k *KademliaNode) RoutingTable() *RoutingTable {
	return k.routingTable
}

// Storage wrapper functions - Add these to your node.go
func (k *KademliaNode) StoreNodeEmbedding(nodeID []byte, embedding []float64) error {
	log.Printf("StoreNodeEmbedding: storing embedding for node %x", nodeID[:8])
	if err := k.storage.StoreNodeEmbedding(nodeID, embedding); err != nil {
		log.Printf("StoreNodeEmbedding: error storing embedding for node %x: %v", nodeID[:8], err)
		return err
	}
	log.Printf("StoreNodeEmbedding: successfully stored embedding for node %x", nodeID[:8])
	return nil
}

func (k *KademliaNode) FindSimilar(queryEmbed []float64, threshold float64, limit int) ([]storage.EmbeddingResult, error) {
	log.Printf("FindSimilar: searching local storage (threshold=%.4f, limit=%d)", threshold, limit)
	results, err := k.storage.FindSimilar(queryEmbed, threshold, limit)
	if err != nil {
		log.Printf("FindSimilar: search error: %v", err)
		return nil, err
	}
	log.Printf("FindSimilar: found %d similar embeddings", len(results))
	return results, nil
}

func (k *KademliaNode) GetID() []byte {
	return k.NodeID
}

func (k *KademliaNode) GetAddress() string {
	return k.PeerID
}

// PrintRoutingTable displays the node's routing table
func (k *KademliaNode) PrintRoutingTable() {
	log.Printf("\n=== ROUTING TABLE FOR NODE %x ===\n", k.NodeID[:8])
	k.routingTable.PrintRoutingTable()
}

// PrintRoutingTableSummary displays a compact view of the routing table
func (k *KademliaNode) PrintRoutingTableSummary() {
	log.Printf("\n=== SUMMARY FOR NODE %x ===\n", k.NodeID[:8])
	k.routingTable.PrintRoutingTableSummary()
}

// PrintPeerInfo displays detailed peer information
func (k *KademliaNode) PrintPeerInfo() {
	log.Printf("\n=== PEER INFO FOR NODE %x ===\n", k.NodeID[:8])
	k.routingTable.PrintPeerInfo()
}

// PrintBucket displays a specific bucket's contents
func (k *KademliaNode) PrintBucket(bucketIndex int) {
	log.Printf("\n=== BUCKET %d FOR NODE %x ===\n", bucketIndex, k.NodeID[:8])
	k.routingTable.PrintBucket(bucketIndex)
}

// FindNode sends a FIND_NODE RPC to the target node to search for searchTargetID
func (k *KademliaNode) FindNode(targetNodeID []byte, searchTargetID []byte) (*types.FindNodeResponse, error) {
	if targetNodeID == nil {
		return nil, fmt.Errorf("target node ID cannot be nil")
	}

	if searchTargetID == nil {
		return nil, fmt.Errorf("search target ID cannot be nil")
	}

	req := &types.FindNodeRequest{
		SenderNodeID: k.NodeID,
		SenderPeerID: k.PeerID,
		TargetID:     searchTargetID,
		Timestamp:    time.Now().UnixNano(),
	}

	log.Printf("FindNode: sending FIND_NODE to %x (searchTarget=%x)", targetNodeID[:8], searchTargetID[:8])

	response, err := k.network.SendFindNode(targetNodeID, req)
	if err != nil {
		log.Printf("FindNode: network.SendFindNode failed for target %x: %v", targetNodeID[:8], err)
		return nil, fmt.Errorf("find node failed: %w", err)
	}

	// Update routing table with nodes from response
	for _, peer := range response.ClosestNodes {
		k.routingTable.Update(peer)
		log.Printf("FindNode: routing table updated with peer %x (peerID=%s)", peer.NodeID[:8], peer.PeerID)
	}

	log.Printf("FindNode: received %d closest nodes from %x", len(response.ClosestNodes), targetNodeID[:8])
	return response, nil
}

// HandleFindNode processes an incoming FIND_NODE request from another node
func (k *KademliaNode) HandleFindNode(req *types.FindNodeRequest) (*types.FindNodeResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("find node request cannot be nil")
	}
	log.Printf("HandleFindNode: received request from %x (peerID=%s) for target %x", req.SenderNodeID[:8], req.SenderPeerID, req.TargetID[:8])

	// Update routing table with sender
	if len(req.SenderNodeID) > 0 {
		senderPeer := types.PeerInfo{
			NodeID: req.SenderNodeID,
			PeerID: req.SenderPeerID,
		}
		k.routingTable.Update(senderPeer)
		log.Printf("HandleFindNode: routing table updated with sender %x", req.SenderNodeID[:8])
	}

	// Find K closest nodes to the target using XOR distance
	closestNodes := k.routingTable.FindClosest(req.TargetID, k.routingTable.K)
	log.Printf("HandleFindNode: found %d closest nodes for target %x", len(closestNodes), req.TargetID[:8])

	response := &types.FindNodeResponse{
		SenderNodeID: k.NodeID,
		SenderPeerID: k.PeerID,
		ClosestNodes: closestNodes,
		Timestamp:    time.Now().UnixNano(),
		Success:      true,
	}

	return response, nil
}

// findNextPeerForSearch finds the best next peer for embedding search routing
func (k *KademliaNode) findNextPeerForSearch(queryEmbed []float64, excludeNodeID []byte) *types.PeerInfo {
	log.Printf("findNextPeerForSearch: hashing embedding for routing decision")
	// Hash the embedding to use for routing
	queryHash := k.hashEmbedding(queryEmbed)

	// Use your routing table's FindClosest method to get candidate peers
	closestPeers := k.routingTable.FindClosest(queryHash, k.routingTable.K)
	log.Printf("findNextPeerForSearch: found %d candidate peers", len(closestPeers))

	// Filter out the source node (the one we want to exclude)
	for _, peer := range closestPeers {
		if string(peer.NodeID) != string(excludeNodeID) {
			log.Printf("findNextPeerForSearch: selected next peer %x (peerID=%s)", peer.NodeID[:8], peer.PeerID)
			return &peer
		}
	}

	// No suitable peers found
	log.Printf("findNextPeerForSearch: no suitable next peer found")
	return nil
}

// IterativeFindNode performs iterative lookup to find and reach target NodeID
func (k *KademliaNode) IterativeFindNode(targetNodeID []byte) (*types.FindNodeResponse, error) {
	if targetNodeID == nil {
		return nil, fmt.Errorf("target NodeID cannot be nil")
	}

	log.Printf("IterativeFindNode: Starting iterative lookup for target: %x", targetNodeID[:8])

	// Track nodes we've already queried to avoid loops
	queriedNodes := make(map[string]bool)
	queriedNodes[string(k.NodeID)] = true // Don't query ourselves

	// Get initial closest nodes from our routing table
	closestNodes := k.routingTable.FindClosest(targetNodeID, k.routingTable.K)
	if len(closestNodes) == 0 {
		log.Printf("IterativeFindNode: no nodes in routing table to start lookup")
		return nil, fmt.Errorf("no nodes in routing table to start lookup")
	}

	log.Printf("IterativeFindNode: starting with %d initial nodes from routing table", len(closestNodes))

	// Track the closest nodes found so far
	allFoundNodes := make(map[string]types.PeerInfo)

	// Add initial nodes
	for _, peer := range closestNodes {
		allFoundNodes[string(peer.NodeID)] = peer
	}

	maxIterations := 10 // Prevent infinite loops
	iteration := 0

	for iteration < maxIterations {
		iteration++
		log.Printf("IterativeFindNode: iteration %d: querying nodes...", iteration)

		// Check if we've reached the target
		if _, exists := allFoundNodes[string(targetNodeID)]; exists {
			log.Printf("IterativeFindNode: target node found in iteration %d", iteration)
			return &types.FindNodeResponse{
				SenderNodeID: targetNodeID,
				SenderPeerID: allFoundNodes[string(targetNodeID)].PeerID,
				ClosestNodes: k.getKClosestNodes(targetNodeID, allFoundNodes),
				Timestamp:    time.Now().UnixNano(),
				Success:      true,
			}, nil
		}

		// Find unqueried nodes to ask
		nodesToQuery := k.selectNodesToQuery(targetNodeID, allFoundNodes, queriedNodes)
		if len(nodesToQuery) == 0 {
			log.Printf("IterativeFindNode: no more nodes to query. lookup complete.")
			break
		}

		log.Printf("IterativeFindNode: querying %d nodes in this iteration", len(nodesToQuery))

		// Query multiple nodes in parallel (simplified sequential for now)
		newNodesFound := false
		for _, nodeToQuery := range nodesToQuery {
			log.Printf("IterativeFindNode: querying node %x", nodeToQuery.NodeID[:8])

			response, err := k.FindNode(nodeToQuery.NodeID, targetNodeID)
			queriedNodes[string(nodeToQuery.NodeID)] = true

			if err != nil {
				log.Printf("IterativeFindNode: failed to query node %x: %v", nodeToQuery.NodeID[:8], err)
				continue
			}

			// Add new nodes from response
			for _, newNode := range response.ClosestNodes {
				if _, exists := allFoundNodes[string(newNode.NodeID)]; !exists {
					allFoundNodes[string(newNode.NodeID)] = newNode
					newNodesFound = true
					log.Printf("IterativeFindNode: found new node %x", newNode.NodeID[:8])
				}
			}
		}

		// If no new nodes found, we're done
		if !newNodesFound {
			log.Printf("IterativeFindNode: no new nodes discovered. lookup complete.")
			break
		}
	}

	// Return the closest nodes we found
	closestFound := k.getKClosestNodes(targetNodeID, allFoundNodes)
	log.Printf("IterativeFindNode: lookup finished, returning %d closest nodes", len(closestFound))

	return &types.FindNodeResponse{
		SenderNodeID: k.NodeID,
		SenderPeerID: k.PeerID,
		ClosestNodes: closestFound,
		Timestamp:    time.Now().UnixNano(),
		Success:      len(closestFound) > 0,
	}, nil
}

// selectNodesToQuery selects the best nodes to query next in iterative lookup
func (k *KademliaNode) selectNodesToQuery(targetID []byte, allNodes map[string]types.PeerInfo, queriedNodes map[string]bool) []types.PeerInfo {
	var candidates []types.PeerInfo

	// Get unqueried nodes
	for nodeIDStr, peer := range allNodes {
		if !queriedNodes[nodeIDStr] {
			candidates = append(candidates, peer)
		}
	}

	// Sort by distance to target
	sort.Slice(candidates, func(i, j int) bool {
		distI := helpers.XORDistance(targetID, candidates[i].NodeID)
		distJ := helpers.XORDistance(targetID, candidates[j].NodeID)
		return distI.Cmp(distJ) < 0
	})

	// Return closest α nodes (typically 3)
	alpha := 3
	if len(candidates) < alpha {
		log.Printf("selectNodesToQuery: returning %d candidates (less than alpha=%d)", len(candidates), alpha)
		return candidates
	}
	log.Printf("selectNodesToQuery: returning alpha=%d closest candidates", alpha)
	return candidates[:alpha]
}

// getKClosestNodes returns K closest nodes from discovered nodes
func (k *KademliaNode) getKClosestNodes(targetID []byte, allNodes map[string]types.PeerInfo) []types.PeerInfo {
	var nodes []types.PeerInfo
	for _, peer := range allNodes {
		nodes = append(nodes, peer)
	}

	// Sort by distance to target
	sort.Slice(nodes, func(i, j int) bool {
		distI := helpers.XORDistance(targetID, nodes[i].NodeID)
		distJ := helpers.XORDistance(targetID, nodes[j].NodeID)
		return distI.Cmp(distJ) < 0
	})

	// Return K closest
	k_value := k.routingTable.K
	if len(nodes) < k_value {
		log.Printf("getKClosestNodes: returning all %d nodes (< K=%d)", len(nodes), k_value)
		return nodes
	}
	log.Printf("getKClosestNodes: returning K=%d closest nodes", k_value)
	return nodes[:k_value]
}

// hashEmbedding converts an embedding vector to a hash for routing decisions
func (k *KademliaNode) hashEmbedding(embedding []float64) []byte {
	log.Printf("hashEmbedding: hashing embedding of length %d", len(embedding))
	data := make([]byte, len(embedding)*8)
	for i, val := range embedding {
		bits := math.Float64bits(val)
		binary.LittleEndian.PutUint64(data[i*8:(i+1)*8], bits)
	}

	hasher := sha1.New()
	hasher.Write(data)
	sum := hasher.Sum(nil)
	log.Printf("hashEmbedding: produced hash %x", sum[:8])
	return sum
}

func (k *KademliaNode) CompleteEmbeddingLookup(queryEmbed []float64) (*types.EmbeddingSearchResponse, error) {
	log.Printf("=== Starting Complete Embedding Lookup ===")
	// STEP 1: Use embedding search to find target NodeID
	log.Printf("Step 1: Finding target NodeID using embedding similarity...")

	// Use FindSimilar - get the best match from the list
	closestNodes, err := k.FindSimilar(queryEmbed, 0.0, k.routingTable.K)
	if err != nil {
		log.Printf("CompleteEmbeddingLookup: failed to find similar embeddings: %v", err)
		return nil, fmt.Errorf("failed to find similar embeddings: %w", err)
	}

	if len(closestNodes) == 0 {
		log.Printf("CompleteEmbeddingLookup: no similar embeddings found in local storage")
		return nil, fmt.Errorf("no similar embeddings found in local storage")
	}

	// Use the best match (first element)
	targetNodeID := closestNodes[0].Key
	log.Printf("Step 1 Complete: Target NodeID found: %x (similarity: %.3f)", targetNodeID[:8], closestNodes[0].Similarity)

	// STEP 2: Use iterative FIND_NODE to route to target
	log.Printf("Step 2: Routing to target NodeID using Kademlia...")
	lookupResponse, err := k.IterativeFindNode(targetNodeID)
	if err != nil {
		log.Printf("CompleteEmbeddingLookup: failed to reach target node: %v", err)
		return nil, fmt.Errorf("failed to reach target node: %w", err)
	}
	log.Printf("Step 2 Complete: Lookup finished")

	// STEP 3: Check if we actually reached the target
	var finalResponse *types.EmbeddingSearchResponse

	if string(lookupResponse.SenderNodeID) == string(targetNodeID) {
		// We reached the target! It responds with the embedding
		log.Printf("Step 3: Successfully reached target node %x", targetNodeID[:8])

		finalResponse = &types.EmbeddingSearchResponse{
			QueryType:    "complete_lookup",
			QueryEmbed:   queryEmbed,
			Depth:        1,
			SourceNodeID: targetNodeID,
			SourcePeerID: lookupResponse.SenderPeerID,
			NextNodeID:   targetNodeID,
			Found:        true,
			FileEmbed:    closestNodes[0].Embedding,
		}
	} else {
		// We got close but didn't reach the exact target
		log.Printf("Step 3: Reached closest available node %x (not exact target %x)", lookupResponse.SenderNodeID[:8], targetNodeID[:8])

		var nextNodeID []byte
		if len(lookupResponse.ClosestNodes) > 0 {
			nextNodeID = lookupResponse.ClosestNodes[0].NodeID
			log.Printf("CompleteEmbeddingLookup: next hop candidate %x", nextNodeID[:8])
		} else {
			nextNodeID = lookupResponse.SenderNodeID
			log.Printf("CompleteEmbeddingLookup: using sender node %x as next hop", nextNodeID[:8])
		}

		finalResponse = &types.EmbeddingSearchResponse{
			QueryType:    "complete_lookup",
			QueryEmbed:   queryEmbed,
			Depth:        1,
			SourceNodeID: lookupResponse.SenderNodeID,
			SourcePeerID: lookupResponse.SenderPeerID,
			NextNodeID:   nextNodeID,
			Found:        false,
			FileEmbed:    nil,
		}
	}

	log.Printf("=== Complete Embedding Lookup Finished ===")
	return finalResponse, nil
}

// HandleEmbeddingSearch processes incoming embedding search requests
func (k *KademliaNode) HandleEmbeddingSearch(req *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("embedding search request cannot be nil")
	}
	log.Printf("HandleEmbeddingSearch: received query (type=%s, depth=%d, resultsCount=%d) from %x", req.QueryType, req.Depth, req.ResultsCount, req.SourceNodeID[:8])

	// Update routing table with source node
	if len(req.SourceNodeID) > 0 {
		sourcePeer := types.PeerInfo{
			NodeID: req.SourceNodeID,
			PeerID: req.SourcePeerID,
		}
		k.routingTable.Update(sourcePeer)
		log.Printf("HandleEmbeddingSearch: routing table updated with source %x", req.SourceNodeID[:8])
	}

	// Find closest node embeddings from local database using FindSimilar
	closestNodes, err := k.FindSimilar(req.QueryEmbed, req.Threshold, req.ResultsCount)
	if err != nil {
		log.Printf("HandleEmbeddingSearch: database search failed: %v", err)
		return nil, fmt.Errorf("database search failed: %w", err)
	}
	log.Printf("HandleEmbeddingSearch: local DB returned %d candidates", len(closestNodes))

	response := &types.EmbeddingSearchResponse{
		QueryType:    req.QueryType,
		QueryEmbed:   req.QueryEmbed,
		Depth:        req.Depth + 1,
		SourceNodeID: k.NodeID,
		SourcePeerID: k.PeerID,
		Found:        len(closestNodes) > 0,
		NextNodeID:   nil,
	}

	// If we found similar embeddings, return the best one
	if len(closestNodes) > 0 {
		response.FileEmbed = closestNodes[0].Embedding
		response.NextNodeID = closestNodes[0].Key
		log.Printf("HandleEmbeddingSearch: returning best match %x (similarity: %.4f)", closestNodes[0].Key[:8], closestNodes[0].Similarity)
		return response, nil
	}

	// If no nodes in database, try to find next node from routing table
	nextPeer := k.findNextPeerForSearch(req.QueryEmbed, req.SourceNodeID)
	if nextPeer != nil {
		response.NextNodeID = nextPeer.NodeID
		log.Printf("HandleEmbeddingSearch: forwarding to next peer %x (peerID=%s)", nextPeer.NodeID[:8], nextPeer.PeerID)
	} else {
		log.Printf("HandleEmbeddingSearch: no next peer found to forward")
	}

	return response, nil
}
