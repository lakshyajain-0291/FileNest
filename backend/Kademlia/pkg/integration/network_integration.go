package integration

import (
	"bytes"
	"database/sql"
	"encoding/hex"
	"final/backend/pkg/helpers"
	"final/backend/pkg/identity"
	"final/backend/pkg/kademlia"
	"final/backend/pkg/types"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	_ "github.com/mattn/go-sqlite3"
)

// Local wrapper type to extend helpers.EmbeddingProcessor
type LocalEmbeddingProcessor struct {
	*helpers.EmbeddingProcessor
}

// NewLocalEmbeddingProcessor creates a wrapper around helpers.EmbeddingProcessor
func NewLocalEmbeddingProcessor() *LocalEmbeddingProcessor {
	return &LocalEmbeddingProcessor{
		EmbeddingProcessor: &helpers.EmbeddingProcessor{},
	}
}

// NetworkIntegrationService handles network layer integration with Kademlia DHT
type NetworkIntegrationService struct {
	kademliaNode       *kademlia.KademliaNode
	embeddingProcessor *LocalEmbeddingProcessor
	hostPeerID         peer.ID
	currentDepth       int
}

// NewNetworkIntegrationService creates a new network integration service
func NewNetworkIntegrationService(node *kademlia.KademliaNode, processor *LocalEmbeddingProcessor, depth int) *NetworkIntegrationService {
	return &NetworkIntegrationService{
		kademliaNode:       node,
		embeddingProcessor: processor,
		hostPeerID:         peer.ID(node.GetAddress()),
		currentDepth:       depth,
	}
}

// ProcessEmbeddingRequest - Main function implementing the requested logic
func (nis *NetworkIntegrationService) ProcessEmbeddingRequest(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Processing embedding request: type=%s, target=%x, source=%s, depth=%d",
		request.QueryType, request.TargetNodeID[:8], request.SourcePeerID, request.Depth)

	// Get current node's ID
	myNodeID := nis.kademliaNode.GetID()

	// Check if target node ID is the same as peer's node ID using bytes.Equal
	if bytes.Equal(request.TargetNodeID, myNodeID) {
		log.Printf("Target node matches current peer - finding next node by cosine similarity")
		return nis.findNextNodeBySimilarity(request)
	} else {
		log.Printf("Target node doesn't match - routing via Kademlia to target %x", request.TargetNodeID[:8])
		return nis.routeViaKademlia(request)
	}
}

// findNextNodeBySimilarity - When target matches current node, find next node by similarity
func (nis *NetworkIntegrationService) findNextNodeBySimilarity(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Finding next node by cosine similarity for embedding")

	storedEmbeddings, err := nis.kademliaNode.FindSimilar(request.QueryEmbed, 0.0, 100)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve stored embeddings: %w", err)
	}

	if len(storedEmbeddings) == 0 {
		return &types.EmbeddingSearchResponse{
			QueryType:    "no_embeddings",
			QueryEmbed:   request.QueryEmbed,
			Depth:        request.Depth,
			SourceNodeID: nis.kademliaNode.GetID(),
			SourcePeerID: nis.kademliaNode.GetAddress(),
			Found:        false,
			NextNodeID:   nil,
			FileEmbed:    nil,
		}, fmt.Errorf("no embeddings found for similarity comparison")
	}

	var bestMatch *EmbeddingResult
	maxSimilarity := -2.0

	for _, stored := range storedEmbeddings {
		similarity, err := nis.embeddingProcessor.CosineSimilarity(request.QueryEmbed, stored.Embedding)
		if err != nil {
			continue
		}

		if similarity > maxSimilarity {
			maxSimilarity = similarity
			bestMatch = &EmbeddingResult{
				NodeID:     stored.Key,
				Embedding:  stored.Embedding,
				Similarity: similarity,
			}
		}
	}

	if bestMatch == nil {
		return &types.EmbeddingSearchResponse{
			QueryType:    "similarity_error",
			QueryEmbed:   request.QueryEmbed,
			Depth:        request.Depth,
			SourceNodeID: nis.kademliaNode.GetID(),
			SourcePeerID: nis.kademliaNode.GetAddress(),
			Found:        false,
			NextNodeID:   nil,
			FileEmbed:    nil,
		}, fmt.Errorf("no valid similarity matches found")
	}

	log.Printf("Found next node by similarity: %x (similarity: %.4f)", bestMatch.NodeID[:8], bestMatch.Similarity)

	return &types.EmbeddingSearchResponse{
		QueryType:    "similarity_match",
		QueryEmbed:   request.QueryEmbed,
		Depth:        request.Depth + 1,
		SourceNodeID: nis.kademliaNode.GetID(),
		SourcePeerID: nis.kademliaNode.GetAddress(),
		Found:        true,
		NextNodeID:   bestMatch.NodeID,
		FileEmbed:    bestMatch.Embedding,
	}, nil
}

// routeViaKademlia - When target doesn't match, route via Kademlia DHT
func (nis *NetworkIntegrationService) routeViaKademlia(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Routing to target node %x via Kademlia", request.TargetNodeID[:8])

	rt := nis.kademliaNode.RoutingTable()
	closestPeers := rt.FindClosest(request.TargetNodeID, rt.K)

	if len(closestPeers) == 0 {
		return &types.EmbeddingSearchResponse{
			QueryType:    "routing_error",
			QueryEmbed:   request.QueryEmbed,
			Depth:        request.Depth,
			SourceNodeID: nis.kademliaNode.GetID(),
			SourcePeerID: nis.kademliaNode.GetAddress(),
			Found:        false,
			NextNodeID:   nil,
			FileEmbed:    nil,
		}, fmt.Errorf("no peers available for routing to target %x", request.TargetNodeID[:8])
	}

	nextHop := closestPeers[0]
	log.Printf("Routing to next hop: %s (node ID: %x)", nextHop.PeerID, nextHop.NodeID[:8])

	return &types.EmbeddingSearchResponse{
		QueryType:    "routed",
		QueryEmbed:   request.QueryEmbed,
		Depth:        request.Depth,
		SourceNodeID: nis.kademliaNode.GetID(),
		SourcePeerID: nis.kademliaNode.GetAddress(),
		Found:        false,
		NextNodeID:   nextHop.NodeID,
		FileEmbed:    nil,
	}, nil
}

// Supporting type for embedding results
type EmbeddingResult struct {
	NodeID     []byte    `json:"node_id"`
	Embedding  []float64 `json:"embedding"`
	Similarity float64   `json:"similarity"`
}

// ========== COMPREHENSIVE WRAPPER FUNCTIONS ==========

// ComprehensiveKademliaHandler - Single wrapper that handles all Kademlia operations
type ComprehensiveKademliaHandler struct {
	node           *kademlia.KademliaNode
	networkService *NetworkIntegrationService
	isInitialized  bool
}

// NewComprehensiveKademliaHandler creates the handler
func NewComprehensiveKademliaHandler() *ComprehensiveKademliaHandler {
	return &ComprehensiveKademliaHandler{
		isInitialized: false,
	}
}

// InitializeNode - Initialize Kademlia node
func (ckh *ComprehensiveKademliaHandler) InitializeNode(peerID, dbPath string) ([]byte,error) {
	if ckh.isInitialized {
		return nil, nil
	}

	// ✅ Convert ONCE at initialization
	nodeID, err := identity.LoadOrCreateNodeID("")
	if err != nil {
		return nil, fmt.Errorf("failed to load or create node ID: %w", err)
	}

	var network kademlia.NetworkInterface
	log.Printf("network integration, nodeLen %v", len(nodeID))
	node, err := kademlia.NewKademliaNode(nodeID, peerID, network, dbPath)
	if err != nil {
		return nodeID, fmt.Errorf("failed to create Kademlia node: %w", err)
	}

	processor := NewLocalEmbeddingProcessor()
	networkService := NewNetworkIntegrationService(node, processor, 0)

	// add code for loading routing table from the database.
	// if the database doesn't exist, it should be created and the routing table should be

	ckh.node = node
	ckh.networkService = networkService
	ckh.isInitialized = true

	log.Printf("Kademlia node initialized: %x", nodeID[:8])

	// --- Routing Table DB Logic ---
	routingDBPath := "routing_table.db"
	var db *sql.DB
	if _, err := os.Stat(routingDBPath); os.IsNotExist(err) {
		db, err = sql.Open("sqlite3", routingDBPath)
		if err != nil {
			return nodeID, fmt.Errorf("failed to open routing table db: %w", err)
		}
		defer db.Close()

		// Create table if not exists
		_, err = db.Exec(`CREATE TABLE IF NOT EXISTS routing_table (
			node_id TEXT PRIMARY KEY,
			peer_id TEXT
		)`)
		if err != nil {
			return nodeID, fmt.Errorf("failed to create routing table: %w", err)
		}
		log.Printf("Created routing_table.db and routing_table table.")
	} else {
		db, err = sql.Open("sqlite3", routingDBPath)
		if err != nil {
			return nodeID, fmt.Errorf("failed to open routing table db: %w", err)
		}
		defer db.Close()
	}

	// Load existing entries into routing table
	rows, err := db.Query("SELECT node_id, peer_id FROM routing_table")
	if err != nil {
		return nodeID, fmt.Errorf("failed to query routing table: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var nodeIDHex, peerID string
		if err := rows.Scan(&nodeIDHex, &peerID); err != nil {
			return nodeID,fmt.Errorf("failed to scan routing table row: %w", err)
		}
		nodeIDBytes, err := hex.DecodeString(nodeIDHex)
		if err != nil {
			log.Printf("Invalid node_id in routing table: %s", nodeIDHex)
			continue
		}
		peerInfo := types.PeerInfo{NodeID: nodeIDBytes, PeerID: peerID}
		node.RoutingTable().Update(peerInfo)
	}

	return nodeID, nil
}

// ProcessEmbeddingRequestWrapper - Main function using []byte throughout
func (ckh *ComprehensiveKademliaHandler) ProcessEmbeddingRequestWrapper(
	queryEmbed []float64,
	targetNodeID []byte, // ✅ []byte parameter
	queryType string,
	threshold float64,
	resultsCount int,
) (*types.EmbeddingSearchResponse, error) {

	if !ckh.isInitialized {
		return nil, fmt.Errorf("node not initialized - call InitializeNode first")
	}

	request := &types.EmbeddingSearchRequest{
		SourceNodeID: ckh.node.GetID(),
		SourcePeerID: ckh.node.GetAddress(),
		QueryEmbed:   queryEmbed,
		Depth:        0,
		QueryType:    queryType,
		Threshold:    threshold,
		ResultsCount: resultsCount,
		TargetNodeID: targetNodeID, // ✅ Direct use - no conversion
	}

	return ckh.networkService.ProcessEmbeddingRequest(request)
}

// HandleIncomingEmbeddingSearch - Handle incoming embedding search requests
func (ckh *ComprehensiveKademliaHandler) HandleIncomingEmbeddingSearch(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	if !ckh.isInitialized {
		return nil, fmt.Errorf("node not initialized")
	}

	return ckh.node.HandleEmbeddingSearch(request)
}

// HandleIncomingFindNode - Handle incoming find node requests
func (ckh *ComprehensiveKademliaHandler) HandleIncomingFindNode(request *types.FindNodeRequest) (*types.FindNodeResponse, error) {
	if !ckh.isInitialized {
		return nil, fmt.Errorf("node not initialized")
	}

	return ckh.node.HandleFindNode(request)
}

// HandleIncomingPing - Handle incoming ping requests
func (ckh *ComprehensiveKademliaHandler) HandleIncomingPing(request *types.PingRequest) (*types.PingResponse, error) {
	if !ckh.isInitialized {
		return nil, fmt.Errorf("node not initialized")
	}

	return ckh.node.HandlePing(request)
}

// StoreEmbedding - Store an embedding using []byte node ID
func (ckh *ComprehensiveKademliaHandler) StoreEmbedding(targetNodeID []byte, embedding []float64) error {
	if !ckh.isInitialized {
		return fmt.Errorf("node not initialized")
	}

	log.Printf("[StoreEmbedding] NodeID length: %d", len(targetNodeID))

	// ✅ Direct use - no conversion
	return ckh.node.StoreNodeEmbedding(targetNodeID, embedding)
}

// CompleteEmbeddingLookup - Perform complete embedding lookup
func (ckh *ComprehensiveKademliaHandler) CompleteEmbeddingLookup(queryEmbed []float64) (*types.EmbeddingSearchResponse, error) {
	if !ckh.isInitialized {
		return nil, fmt.Errorf("node not initialized")
	}

	log.Printf("[CompleteEmbeddingLookup] NodeID length: %d", len(ckh.node.GetID()))

	return ckh.node.CompleteEmbeddingLookup(queryEmbed)
}

// IterativeFindNode - Perform iterative find node lookup using []byte
func (ckh *ComprehensiveKademliaHandler) IterativeFindNode(targetNodeID []byte) (*types.FindNodeResponse, error) {
	if !ckh.isInitialized {
		return nil, fmt.Errorf("node not initialized")
	}

	log.Printf("[IterativeFindNode] NodeID length: %d", len(ckh.node.GetID()))

	// ✅ Direct use - no conversion
	return ckh.node.IterativeFindNode(targetNodeID)
}

// GetNodeStatistics - Get node statistics
func (ckh *ComprehensiveKademliaHandler) GetNodeStatistics() map[string]interface{} {
	if !ckh.isInitialized {
		return map[string]interface{}{
			"initialized": false,
			"error":       "node not initialized",
		}
	}
	log.Printf("[GetNodeStatistics] NodeID: %v", string(ckh.node.GetID()))
	log.Printf("[GetNodeStatistics] NodeID length: %d", len(ckh.node.GetID()))

	rt := ckh.node.RoutingTable()
	stats := map[string]interface{}{
		"node_id":       fmt.Sprintf("%x", ckh.node.GetID()), // !!! was an [:8] here.
		"peer_id":       ckh.node.GetAddress(),
		"routing_peers": len(rt.FindClosest(ckh.node.GetID(), rt.K)),
		"initialized":   ckh.isInitialized,
		"timestamp":     time.Now().Unix(),
	}
	log.Printf("Node Stats: %+v", stats)
	return stats
}

// AddPeerToRoutingTable adds a peer directly to the Kademlia routing table
func (ckh *ComprehensiveKademliaHandler) AddPeerToRoutingTable(peer types.PeerInfo) error {
	if !ckh.isInitialized {
		return fmt.Errorf("kademlia handler not initialized")
	}

	if ckh.node == nil {
		return fmt.Errorf("kademlia node is nil")
	}

	log.Printf("[AddPeerToRoutingTable] NodeID length: %d", len(ckh.node.GetID()))

	routingTable := ckh.node.RoutingTable()
	if routingTable == nil {
		return fmt.Errorf("routing table is nil")
	}

	// Call Update method
	routingTable.Update(peer)

	log.Printf("Added peer to routing table: NodeID=%x, PeerID=%s",
		peer.NodeID, peer.PeerID)

	return nil
}

// GetRoutingInfo - Get routing table information
func (ckh *ComprehensiveKademliaHandler) GetRoutingInfo() []types.PeerInfo {
	if !ckh.isInitialized || ckh.node == nil {
		return nil
	}

	routingTable := ckh.node.RoutingTable()
	if routingTable == nil {
		return nil
	}

	return routingTable.GetNodes()
}

// ========== ADDITIONAL BATCH AND VALIDATION FUNCTIONS ==========

// ProcessBatchEmbeddingRequests processes multiple embedding requests
func (nis *NetworkIntegrationService) ProcessBatchEmbeddingRequests(requests []*types.EmbeddingSearchRequest) ([]*types.EmbeddingSearchResponse, error) {
	responses := make([]*types.EmbeddingSearchResponse, 0, len(requests))

	for i, request := range requests {
		log.Printf("Processing batch request %d/%d", i+1, len(requests))

		response, err := nis.ProcessEmbeddingRequest(request)
		if err != nil {
			log.Printf("Batch request %d failed: %v", i+1, err)
			errorResponse := &types.EmbeddingSearchResponse{
				QueryType:    "batch_error",
				QueryEmbed:   request.QueryEmbed,
				Depth:        request.Depth,
				SourceNodeID: nis.kademliaNode.GetID(),
				SourcePeerID: nis.kademliaNode.GetAddress(),
				Found:        false,
				NextNodeID:   nil,
				FileEmbed:    nil,
			}
			responses = append(responses, errorResponse)
		} else {
			responses = append(responses, response)
		}
	}

	return responses, nil
}

// GetNetworkIntegrationStats returns statistics about network integration
func (nis *NetworkIntegrationService) GetNetworkIntegrationStats() map[string]interface{} {
	rt := nis.kademliaNode.RoutingTable()
	return map[string]interface{}{
		"peer_id":        nis.hostPeerID.String(),
		"node_id":        fmt.Sprintf("%x", nis.kademliaNode.GetID()[:8]),
		"current_depth":  nis.currentDepth,
		"contacts_count": len(rt.FindClosest(nis.kademliaNode.GetID(), rt.K)),
		"timestamp":      time.Now().Unix(),
		"is_d4_node":     nis.currentDepth >= 4,
	}
}

// ValidateEmbeddingRequest validates incoming embedding requests
func (nis *NetworkIntegrationService) ValidateEmbeddingRequest(request *types.EmbeddingSearchRequest) error {
	if len(request.TargetNodeID) == 0 {
		return fmt.Errorf("target node ID cannot be empty")
	}

	if len(request.QueryEmbed) == 0 {
		return fmt.Errorf("embedding vector cannot be empty")
	}

	if request.QueryType == "" {
		return fmt.Errorf("query type cannot be empty")
	}

	if request.Threshold < 0 || request.Threshold > 1 {
		return fmt.Errorf("threshold must be between 0 and 1")
	}

	return nil
}

// ========== HELPER FUNCTIONS ==========

// func ParseBootstrapAddr(addr string) (peer.AddrInfo, error) {
// 	maddr, err := peer.AddrInfoFromString(addr)
// 	if err != nil {
// 		return peer.AddrInfo{}, errors.New("invalid bootstrap node multiaddr")
// 	}
// 	return *maddr, nil
// }

// func XORDistance(a, b []byte) *big.Int {
// 	if len(a) != len(b) {
// 		panic("IDs must be the same length 3")
// 	}
// 	dist := make([]byte, len(a))
// 	for i := range a {
// 		dist[i] = a[i] ^ b[i]
// 	}
// 	return new(big.Int).SetBytes(dist)
// }

// func BucketIndex(selfID, otherID []byte) int {
// 	if len(selfID) != len(otherID) {
// 		panic("IDs must be the same length 4")
// 	}
// 	for byteIndex := range selfID {
// 		xorByte := selfID[byteIndex] ^ otherID[byteIndex]
// 		if xorByte != 0 {
// 			for bitPos := range 8 {
// 				if (xorByte & (0x80 >> bitPos)) != 0 {
// 					return (len(selfID)-byteIndex-1)*8 + (7 - bitPos)
// 				}
// 			}
// 		}
// 	}
// 	return -1
// }

// func RandomNodeID() []byte {
// 	id := make([]byte, identity.NodeIDBytes)
// 	if _, err := rand.Read(id); err != nil {
// 		log.Fatalf("failed to generate random NodeID: %v", err)
// 	}
// 	return id
// }

// AddPeerToRoutingTable adds a peer directly to the Kademlia routing table
// func (ckh *ComprehensiveKademliaHandler) AddPeerToRoutingTable(peer types.PeerInfo) error {
// 	if !ckh.isInitialized {
// 		return fmt.Errorf("kademlia handler not initialized")
// 	}

// 	if ckh.node == nil {
// 		return fmt.Errorf("kademlia node is nil")
// 	}

// 	// ping the peerid using peer.send
// 	// reqJson := network.PingHandler([]byte(peer.PeerID))
// 	// Use getter method instead of direct access
// 	routingTable := ckh.node.RoutingTable()
// 	if routingTable == nil {
// 		return fmt.Errorf("routing table is nil")
// 	}

// 	// Call Update method
// 	routingTable.Update(peer)

// 	log.Printf("Added peer to routing table: NodeID=%x, PeerID=%s",
// 		peer.NodeID, peer.PeerID)

// 	return nil
// }

// BootstrapFromRelayNetwork bootstraps Kademlia using relay network peers
// func (ckh *ComprehensiveKademliaHandler) BootstrapFromRelayNetwork(ctx context.Context, relayAddrs []string) error {
// 	if !ckh.isInitialized {
// 		return fmt.Errorf("kademlia handler not initialized")
// 	}

// 	successfulConnections := 0

// 	for _, addr := range relayAddrs {
// 		// Extract peer ID from multiaddr
// 		parts := strings.Split(addr, "/")
// 		if len(parts) < 2 {
// 			log.Printf("Invalid multiaddr format: %s", addr)
// 			continue
// 		}

// 		peerIDStr := parts[len(parts)-1]

// 		// Convert to node ID
// 		nodeID := helpers.HashNodeIDFromString(peerIDStr)

// 		// Create peer info
// 		peerInfo := types.PeerInfo{
// 			NodeID: nodeID,
// 			PeerID: peerIDStr,
// 		}

// 		// Add to routing table
// 		err := ckh.AddPeerToRoutingTable(peerInfo)
// 		if err != nil {
// 			log.Printf("Failed to add relay peer %s: %v", peerIDStr[:12]+"...", err)
// 			continue
// 		}

// 		successfulConnections++
// 		log.Printf("✓ Added relay peer to Kademlia routing table: %s", peerIDStr[:12]+"...")
// 	}

// 	if successfulConnections == 0 {
// 		return fmt.Errorf("failed to bootstrap from any relay addresses")
// 	}

// 	// Perform iterative lookup to discover more peers
// 	log.Println("🔍 Performing iterative lookup for peer discovery...")
// 	_, err := ckh.IterativeFindNode(ckh.node.NodeID)
// 	if err != nil {
// 		log.Printf("Warning: Iterative lookup failed: %v", err)
// 	} else {
// 		log.Println("✓ Peer discovery completed")
// 	}

// 	log.Printf("✓ Bootstrap completed with %d relay peers added", successfulConnections)
// 	return nil
// }

// GetRoutingTableSize returns the number of peers in routing table
// func (ckh *ComprehensiveKademliaHandler) GetRoutingTableSize() int {
// 	if !ckh.isInitialized || ckh.node == nil {
// 		return 0
// 	}

// 	allPeers := ckh.node.RoutingTable.GetNodes()
// 	return len(allPeers)
// }
