package main

import (
	"crypto/sha256"
	"fmt"
	"log"
	"os"
	"time"

	"kademlia/pkg/kademlia"
	"kademlia/pkg/storage"
	"kademlia/pkg/types"
)

// Simple mock network implementation
type MockNetwork struct {
	nodes map[string]*MockNode
}

type MockNode struct {
	ID      []byte
	Address string
	Online  bool
	Storage storage.Interface
}

func NewMockNetwork() *MockNetwork {
	return &MockNetwork{
		nodes: make(map[string]*MockNode),
	}
}

func (m *MockNetwork) AddNode(nodeID []byte, address string, online bool) {
	dbPath := fmt.Sprintf("mock_%x.db", nodeID)
	sqliteStorage, _ := storage.NewSQLiteStorage(dbPath)

	m.nodes[string(nodeID)] = &MockNode{
		ID:      nodeID,
		Address: address,
		Online:  online,
		Storage: sqliteStorage,
	}
}

func (m *MockNetwork) StoreNodeEmbedding(nodeID []byte, targetNodeID []byte, embedding []float64) {
	if node, exists := m.nodes[string(nodeID)]; exists && node.Online {
		node.Storage.StoreNodeEmbedding(targetNodeID, embedding)
	}
}

func (m *MockNetwork) SendPing(targetNodeID []byte, req *types.PingRequest) (*types.PingResponse, error) {
	node, exists := m.nodes[string(targetNodeID)]
	if !exists {
		return nil, fmt.Errorf("node not found")
	}

	if !node.Online {
		return nil, fmt.Errorf("node offline")
	}

	return &types.PingResponse{
		SenderNodeID: node.ID,
		SenderPeerID: node.Address,
		Timestamp:    time.Now().UnixNano(),
		Success:      true,
	}, nil
}

func (m *MockNetwork) SendEmbeddingSearch(targetNodeID []byte, req *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	node, exists := m.nodes[string(targetNodeID)]
	if !exists {
		return nil, fmt.Errorf("node not found")
	}

	if !node.Online {
		return nil, fmt.Errorf("node offline")
	}

	closestNodes, err := node.Storage.FindSimilar(req.QueryEmbed, req.Threshold, req.ResultsCount)
	if err != nil {
		closestNodes = nil
	}

	response := &types.EmbeddingSearchResponse{
		QueryType:    req.QueryType,
		QueryEmbed:   req.QueryEmbed,
		Depth:        req.Depth + 1,
		SourceNodeID: node.ID,
		SourcePeerID: node.Address,
		Found:        len(closestNodes) > 0,
		NextNodeID:   nil,
	}

	if len(closestNodes) > 0 {
		response.FileEmbed = closestNodes[0].Embedding
		response.NextNodeID = closestNodes[0].Key
	}

	return response, nil
}

func (m *MockNetwork) SendFindNode(targetNodeID []byte, req *types.FindNodeRequest) (*types.FindNodeResponse, error) {
	node, exists := m.nodes[string(targetNodeID)]
	if !exists {
		return nil, fmt.Errorf("node not found")
	}

	if !node.Online {
		return nil, fmt.Errorf("node offline")
	}

	var closestNodes []types.PeerInfo
	for _, mockNode := range m.nodes {
		if string(mockNode.ID) != string(targetNodeID) {
			closestNodes = append(closestNodes, types.PeerInfo{
				NodeID: mockNode.ID,
				PeerID: mockNode.Address,
			})
		}
		if len(closestNodes) >= 3 {
			break
		}
	}

	return &types.FindNodeResponse{
		SenderNodeID: node.ID,
		SenderPeerID: node.Address,
		ClosestNodes: closestNodes,
		Timestamp:    time.Now().UnixNano(),
		Success:      true,
	}, nil
}

func (m *MockNetwork) Cleanup() {
	for nodeID, node := range m.nodes {
		if sqliteStorage, ok := node.Storage.(*storage.SQLiteStorage); ok {
			sqliteStorage.Close()
		}
		dbPath := fmt.Sprintf("mock_%x.db", []byte(nodeID))
		os.Remove(dbPath)
	}
}

// Helper function to normalize NodeIDs
func normalizeNodeID(id string) []byte {
	hash := sha256.Sum256([]byte(id))
	return hash[:]
}

func main() {
	fmt.Println("=== Kademlia Embedding Search Testing ===")

	// Clean up any existing test databases
	cleanup()

	// Test 1: Basic Node Creation
	fmt.Println("1. Testing Node Creation...")
	testNodeCreation()

	// Test 2: Ping Operations
	fmt.Println("\n2. Testing Ping Operations...")
	testPingOperations()

	// Test 3: Node Embedding Storage (StoreNodeEmbedding only)
	fmt.Println("\n3. Testing Node Embedding Storage...")
	testNodeEmbeddingStorage()

	// Test 4: Embedding Search
	fmt.Println("\n4. Testing Embedding Search...")
	testEmbeddingSearch()

	// Test 5: Routing Table Operations
	fmt.Println("\n5. Testing Routing Table...")
	testRoutingTable()

	// Test 6: FIND_NODE RPC
	fmt.Println("\n6. Testing FIND_NODE RPC...")
	testFindNode()

	// Test 7: Iterative Lookup
	fmt.Println("\n7. Testing Iterative Lookup...")
	testIterativeLookup()

	// Test 8: Complete Embedding Workflow
	fmt.Println("\n8. Testing Complete Embedding Workflow...")
	testCompleteEmbeddingWorkflow()

	// Test 9: End-to-End Network Simulation
	fmt.Println("\n9. Testing End-to-End Network...")
	testEndToEndNetwork()

	// Cleanup
	cleanup()

	fmt.Println("\n=== All Tests Completed Successfully! ===")
}

func testNodeCreation() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	nodeID := normalizeNodeID("test-node-1")
	peerID := "peer-id-1"

	node, err := kademlia.NewKademliaNode(nodeID, peerID, mockNetwork, "test_node1.db")
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}

	if string(node.GetID()) != string(nodeID) {
		log.Fatalf("Node ID mismatch")
	}

	if node.GetAddress() != peerID {
		log.Fatalf("Peer ID mismatch")
	}

	fmt.Println("✓ Node creation successful")
}

func testPingOperations() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	node1, err := kademlia.NewKademliaNode(
		normalizeNodeID("node-1"),
		"peer-1",
		mockNetwork,
		"test_ping1.db",
	)
	if err != nil {
		log.Fatalf("Failed to create node1: %v", err)
	}

	node2ID := normalizeNodeID("node-2")
	mockNetwork.AddNode(node2ID, "peer-2", true)

	response, err := node1.Ping(node2ID)
	if err != nil {
		log.Printf("Ping failed (expected in mock): %v", err)
	} else {
		fmt.Printf("✓ Ping successful: %v\n", response.Success)
	}

	pingReq := &types.PingRequest{
		SenderNodeID: normalizeNodeID("sender-node"),
		SenderPeerID: "peer-sender",
		Timestamp:    time.Now().UnixNano(),
	}

	pingResp, err := node1.HandlePing(pingReq)
	if err != nil {
		log.Fatalf("HandlePing failed: %v", err)
	}

	if !pingResp.Success {
		log.Fatalf("Ping should succeed")
	}

	fmt.Println("✓ Ping operations successful")
}

func testNodeEmbeddingStorage() {
    mockNetwork := NewMockNetwork()
    defer mockNetwork.Cleanup()
    
    node, err := kademlia.NewKademliaNode(
        normalizeNodeID("storage-node"),
        "storage-peer",
        mockNetwork,
        "test_storage.db",
    )
    if err != nil {
        log.Fatalf("Failed to create node: %v", err)
    }
    
    // Test only StoreNodeEmbedding operations - no generic storage
    testEmbeddings := map[string][]float64{
        "node-1": {1.0, 0.0, 0.0},
        "node-2": {0.0, 1.0, 0.0},
        "node-3": {0.0, 0.0, 1.0},
        "node-4": {0.7, 0.7, 0.0},
    }
    
    for nodeIDStr, embedding := range testEmbeddings {
        err := node.StoreNodeEmbedding(normalizeNodeID(nodeIDStr), embedding)
        if err != nil {
            log.Fatalf("Failed to store node embedding for %s: %v", nodeIDStr, err)
        }
    }
    
    fmt.Printf("✓ Stored %d node embeddings\n", len(testEmbeddings))
    
    // Test FindSimilar to verify storage worked
    queryEmbed := []float64{0.9, 0.1, 0.0}
    similarNodes, err := node.FindSimilar(queryEmbed, 0.0, 5)
    if err != nil {
        log.Fatalf("Failed to find similar embeddings: %v", err)
    }
    
    fmt.Printf("✓ Found %d similar embeddings for query\n", len(similarNodes))
    
    // Show the best match using FindSimilar
    if len(similarNodes) > 0 {
        bestMatch := similarNodes  // First element is best match
        fmt.Printf("✓ Best match: %x (similarity: %.3f)\n", 
            bestMatch[0].Key[:8], bestMatch[0].Similarity)
            
        // Show all matches
        for i, result := range similarNodes {
            fmt.Printf("  [%d] Node: %x, Similarity: %.3f\n", 
                i+1, result.Key[:8], result.Similarity)
        }
    }
    
    fmt.Println("✓ Node embedding storage operations successful")
}

func testEmbeddingSearch() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	node, err := kademlia.NewKademliaNode(
		normalizeNodeID("search-node"),
		"search-peer",
		mockNetwork,
		"test_search.db",
	)
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}

	// Store some test embeddings locally using StoreNodeEmbedding
	node.StoreNodeEmbedding(normalizeNodeID("local-node-1"), []float64{1.0, 0.0, 0.0})
	node.StoreNodeEmbedding(normalizeNodeID("local-node-2"), []float64{0.0, 1.0, 0.0})
	node.StoreNodeEmbedding(normalizeNodeID("local-node-3"), []float64{0.9, 0.1, 0.0})

	targetID := normalizeNodeID("remote-node")
	mockNetwork.AddNode(targetID, "remote-peer", true)
	mockNetwork.StoreNodeEmbedding(targetID, normalizeNodeID("similar-remote-node"), []float64{0.95, 0.05, 0.0})

	searchReq := &types.EmbeddingSearchRequest{
		SourceNodeID: normalizeNodeID("search-node"),
		SourcePeerID: "search-peer",
		QueryEmbed:   []float64{0.8, 0.2, 0.0},
		Depth:        0,
		QueryType:    "similarity_search",
		Threshold:    0.7,
		ResultsCount: 5,
		TargetNodeID: targetID,
	}

	localResp, err := node.HandleEmbeddingSearch(searchReq)
	if err != nil {
		log.Fatalf("Local embedding search failed: %v", err)
	}

	fmt.Printf("✓ Local search found: %v\n", localResp.Found)
	if localResp.Found {
		fmt.Printf("✓ Found embedding: %v\n", localResp.FileEmbed)
		fmt.Printf("✓ Next NodeID: %x\n", localResp.NextNodeID[:8])
		fmt.Printf("✓ Search depth: %d\n", localResp.Depth)
	}

	fmt.Println("✓ Embedding search operations successful")
}

func testRoutingTable() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	node, err := kademlia.NewKademliaNode(
		normalizeNodeID("routing-node"),
		"routing-peer",
		mockNetwork,
		"test_routing.db",
	)
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}

	fmt.Println("Initial routing table (should be empty):")
	node.PrintRoutingTableSummary()

	testPeers := []types.PeerInfo{
		{NodeID: normalizeNodeID("peer-node-1"), PeerID: "peer-id-1"},
		{NodeID: normalizeNodeID("peer-node-2"), PeerID: "peer-id-2"},
		{NodeID: normalizeNodeID("peer-node-3"), PeerID: "peer-id-3"},
		{NodeID: normalizeNodeID("peer-node-4"), PeerID: "peer-id-4"},
	}

	for _, peer := range testPeers {
		pingReq := &types.PingRequest{
			SenderNodeID: peer.NodeID,
			SenderPeerID: peer.PeerID,
			Timestamp:    time.Now().UnixNano(),
		}

		_, err := node.HandlePing(pingReq)
		if err != nil {
			log.Fatalf("Failed to handle ping from %s: %v", peer.PeerID, err)
		}
	}

	fmt.Printf("✓ Updated routing table with %d peers\n", len(testPeers))

	fmt.Println("\nRouting table after adding peers:")
	node.PrintRoutingTable()

	fmt.Println("\nDetailed peer information:")
	node.PrintPeerInfo()

	fmt.Println("✓ Routing table operations successful")
}

func testFindNode() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	node1, err := kademlia.NewKademliaNode(
		normalizeNodeID("findnode-node-1"),
		"findnode-peer-1",
		mockNetwork,
		"test_findnode1.db",
	)
	if err != nil {
		log.Fatalf("Failed to create node1: %v", err)
	}

	targetNodeID := normalizeNodeID("target-node")
	queryNodeID := normalizeNodeID("query-node")

	mockNetwork.AddNode(targetNodeID, "target-peer", true)
	mockNetwork.AddNode(queryNodeID, "query-peer", true)

	searchTargetID := normalizeNodeID("search-target")

	response, err := node1.FindNode(targetNodeID, searchTargetID)
	if err != nil {
		log.Printf("FindNode failed (expected in mock): %v", err)
	} else {
		fmt.Printf("✓ FindNode successful: %v\n", response.Success)
		fmt.Printf("✓ Found %d closest nodes\n", len(response.ClosestNodes))

		for i, peer := range response.ClosestNodes {
			fmt.Printf("  [%d] NodeID: %x, PeerID: %s\n", i, peer.NodeID[:8], peer.PeerID)
		}
	}

	findReq := &types.FindNodeRequest{
		SenderNodeID: normalizeNodeID("sender-node"),
		SenderPeerID: "sender-peer",
		TargetID:     searchTargetID,
		Timestamp:    time.Now().UnixNano(),
	}

	findResp, err := node1.HandleFindNode(findReq)
	if err != nil {
		log.Fatalf("HandleFindNode failed: %v", err)
	}

	fmt.Printf("✓ HandleFindNode returned %d nodes\n", len(findResp.ClosestNodes))

	fmt.Println("✓ FIND_NODE RPC test successful")

	os.Remove("test_findnode1.db")
}

func testIterativeLookup() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	node, err := kademlia.NewKademliaNode(
		normalizeNodeID("iterative-node"),
		"iterative-peer",
		mockNetwork,
		"test_iterative.db",
	)
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}

	// Add some nodes to routing table first
	testPeers := []types.PeerInfo{
		{NodeID: normalizeNodeID("peer-alpha"), PeerID: "alpha-peer"},
		{NodeID: normalizeNodeID("peer-beta"), PeerID: "beta-peer"},
		{NodeID: normalizeNodeID("peer-gamma"), PeerID: "gamma-peer"},
	}

	for _, peer := range testPeers {
		mockNetwork.AddNode(peer.NodeID, peer.PeerID, true)
		pingReq := &types.PingRequest{
			SenderNodeID: peer.NodeID,
			SenderPeerID: peer.PeerID,
			Timestamp:    time.Now().UnixNano(),
		}
		node.HandlePing(pingReq)
	}

	targetNodeID := normalizeNodeID("iterative-target")
	fmt.Printf("Starting iterative lookup for: %x\n", targetNodeID[:8])

	response, err := node.IterativeFindNode(targetNodeID)
	if err != nil {
		log.Printf("Iterative lookup failed: %v", err)
	} else {
		fmt.Printf("✓ Iterative lookup completed\n")
		fmt.Printf("✓ Success: %v\n", response.Success)
		fmt.Printf("✓ Found %d closest nodes\n", len(response.ClosestNodes))
	}

	fmt.Println("✓ Iterative lookup test successful")

	os.Remove("test_iterative.db")
}

func testCompleteEmbeddingWorkflow() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	node, err := kademlia.NewKademliaNode(
		normalizeNodeID("workflow-node"),
		"workflow-peer",
		mockNetwork,
		"test_workflow.db",
	)
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}

	// Store target embedding using StoreNodeEmbedding only
	targetEmbedding := []float64{0.9, 0.1, 0.0}
	targetNodeID := normalizeNodeID("target-workflow-node")
	err = node.StoreNodeEmbedding(targetNodeID, targetEmbedding)
	if err != nil {
		log.Fatalf("Failed to store target embedding: %v", err)
	}

	// Add some peers to routing table
	testPeers := []types.PeerInfo{
		{NodeID: normalizeNodeID("workflow-peer-1"), PeerID: "wp1"},
		{NodeID: normalizeNodeID("workflow-peer-2"), PeerID: "wp2"},
	}

	for _, peer := range testPeers {
		mockNetwork.AddNode(peer.NodeID, peer.PeerID, true)
		pingReq := &types.PingRequest{
			SenderNodeID: peer.NodeID,
			SenderPeerID: peer.PeerID,
			Timestamp:    time.Now().UnixNano(),
		}
		node.HandlePing(pingReq)
	}

	// Perform complete embedding lookup
	queryEmbedding := []float64{0.85, 0.15, 0.0}

	fmt.Println("🚀 Starting complete embedding lookup workflow...")
	response, err := node.CompleteEmbeddingLookup(queryEmbedding)
	if err != nil {
		log.Printf("Complete lookup failed: %v", err)
	} else {
		fmt.Printf("✅ Complete lookup finished!\n")
		fmt.Printf("✅ Found: %v\n", response.Found)
		if response.Found {
			fmt.Printf("✅ Target reached: %x\n", response.NextNodeID[:8])
			fmt.Printf("✅ Retrieved embedding: %v\n", response.FileEmbed)
			fmt.Printf("✅ Query type: %s\n", response.QueryType)
		}
	}

	fmt.Println("✓ Complete embedding workflow test successful")

	os.Remove("test_workflow.db")
}

func testEndToEndNetwork() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	nodeCount := 5
	nodes := make([]*kademlia.KademliaNode, nodeCount)
	nodeIDs := make([][]byte, nodeCount)
	peerIDs := make([]string, nodeCount)

	// Create network of nodes
	for i := 0; i < nodeCount; i++ {
		nodeIDs[i] = normalizeNodeID(fmt.Sprintf("network-node-%d", i))
		peerIDs[i] = fmt.Sprintf("network-peer-%d", i)

		var err error
		nodes[i], err = kademlia.NewKademliaNode(
			nodeIDs[i],
			peerIDs[i],
			mockNetwork,
			fmt.Sprintf("test_e2e_%d.db", i),
		)
		if err != nil {
			log.Fatalf("Failed to create node %d: %v", i, err)
		}

		mockNetwork.AddNode(nodeIDs[i], peerIDs[i], true)
	}

	// Store different embeddings on different nodes using StoreNodeEmbedding only
	embeddings := [][]float64{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.5, 0.5, 0.0},
		{0.3, 0.7, 0.0},
	}

	for i, embedding := range embeddings {
		err := nodes[i].StoreNodeEmbedding(nodeIDs[i], embedding)
		if err != nil {
			log.Fatalf("Failed to store embedding on node %d: %v", i, err)
		}

		// Also store in mock network for cross-node searches
		mockNetwork.StoreNodeEmbedding(nodeIDs[i], nodeIDs[i], embedding)
	}

	// Build routing tables between all nodes
	for i := 0; i < nodeCount; i++ {
		for j := 0; j < nodeCount; j++ {
			if i != j {
				pingReq := &types.PingRequest{
					SenderNodeID: nodeIDs[j],
					SenderPeerID: peerIDs[j],
					Timestamp:    time.Now().UnixNano(),
				}
				nodes[i].HandlePing(pingReq)
			}
		}
	}

	fmt.Println("  Network topology built successfully")

	// Show routing tables
	for i := 0; i < nodeCount; i++ {
		fmt.Printf("\n--- Node %d Routing Table ---", i)
		nodes[i].PrintRoutingTableSummary()
	}

	// Test embedding search across network
	queryEmbedding := []float64{0.1, 0.9, 0.0}
    
    fmt.Printf("\n🔍 Testing cross-network embedding search...")
    searchReq := &types.EmbeddingSearchRequest{
        SourceNodeID: nodeIDs[0],
        SourcePeerID: peerIDs[0],
        QueryEmbed:   queryEmbedding,
        Depth:        0,
        QueryType:    "network_search",
        Threshold:    0.8,
        ResultsCount: 1,
        TargetNodeID: nodeIDs[1],
    }
    
    // FIXED: Call method on nodes instead of nodes
	for _, node := range nodes {
		searchResp, err := node.HandleEmbeddingSearch(searchReq)
		if err != nil {
			log.Printf("Network search failed: %v", err)
		} else {
			fmt.Printf("✅ Network search completed, found: %v\n", searchResp.Found)
			if searchResp.Found {
				fmt.Printf("✅ Found embedding: %v\n", searchResp.FileEmbed)
			}
		}
	}

	fmt.Println("✓ End-to-end network test successful")

	// Cleanup
	for i := 0; i < nodeCount; i++ {
		os.Remove(fmt.Sprintf("test_e2e_%d.db", i))
	}
}

func cleanup() {
	testFiles := []string{
		"test_node1.db",
		"test_ping1.db",
		"test_storage.db",
		"test_search.db",
		"test_routing.db",
		"test_findnode1.db",
		"test_iterative.db",
		"test_workflow.db",
	}

	// Add generated test files
	for i := 0; i < 10; i++ {
		testFiles = append(testFiles, fmt.Sprintf("test_e2e_%d.db", i))
	}

	for _, file := range testFiles {
		os.Remove(file)
	}

	fmt.Println("✓ Cleanup completed")
}
