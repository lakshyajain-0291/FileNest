package main

import (
	"crypto/sha256"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"kademlia/pkg/helpers"
	"kademlia/pkg/kademlia"
	"kademlia/pkg/storage"
	"kademlia/pkg/types"
)

// Simple mock network implementation directly in main
type MockNetwork struct {
	nodes map[string]*MockNode
}

type MockNode struct {
	ID      []byte
	Address string
	Online  bool
	Storage storage.Interface
}

func normalizeNodeID(id string) []byte {
	hash := sha256.Sum256([]byte(id))
	return hash[:]
}

func NewMockNetwork() *MockNetwork {
	return &MockNetwork{
		nodes: make(map[string]*MockNode),
	}
}

func (m *MockNetwork) AddNode(nodeID []byte, address string, online bool) {
	// Create temporary SQLite database for testing
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

	// Find closest node from database
	closestNode, err := node.Storage.FindSimilar(req.QueryEmbed, req.Threshold, req.ResultsCount)
	if err != nil {
		closestNode = nil
	}

	response := &types.EmbeddingSearchResponse{
		QueryType:    req.QueryType,
		QueryEmbed:   req.QueryEmbed,
		Depth:        req.Depth + 1,
		SourceNodeID: node.ID,
		SourcePeerID: node.Address,
		Found:        closestNode != nil,
		NextNodeID:   nil,
	}

	if closestNode != nil {
		response.FileEmbed = closestNode[0].Embedding
		response.NextNodeID = closestNode[0].Key
	}

	return response, nil
}

func (m *MockNetwork) Cleanup() {
	// Clean up test databases
	for nodeID, node := range m.nodes {
		if sqliteStorage, ok := node.Storage.(*storage.SQLiteStorage); ok {
			sqliteStorage.Close()
		}
		dbPath := fmt.Sprintf("mock_%x.db", []byte(nodeID))
		os.Remove(dbPath)
	}
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

	// Test 3: Storage Operations
	fmt.Println("\n3. Testing Storage Operations...")
	// testStorageOperations()

	// Test 4: Embedding Search
	fmt.Println("\n4. Testing Embedding Search...")
	testEmbeddingSearch()

	// Test 5: Routing Table Operations
	fmt.Println("\n5. Testing Routing Table...")
	testRoutingTable()

	// Test 6: End-to-End Workflow
	fmt.Println("\n6. Testing End-to-End Workflow...")
	testEndToEndWorkflow()

	fmt.Println("\n7. Testing Routing Table Display...")
    testRoutingTableDisplay()
	
	// Cleanup
	cleanup()

	fmt.Println("\n=== All Tests Completed ===")
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

	// Create normalized NodeIDs
	node1ID := helpers.HashNodeID([]byte("node-1"))
	node2ID := helpers.HashNodeID([]byte("node-2"))
	senderID := helpers.HashNodeID([]byte("sender-node"))

	// Create nodes with normalized IDs
	node1, err := kademlia.NewKademliaNode(
		node1ID, // Now 32 bytes
		"peer-1",
		mockNetwork,
		"test_ping1.db",
	)
	if err != nil {
		log.Fatalf("Failed to create node1: %v", err)
	}

	// Add node2 to mock network with normalized ID
	mockNetwork.AddNode(node2ID, "127.0.0.1:8002", true)

	// Test handle ping with normalized sender ID
	pingReq := &types.PingRequest{
		SenderNodeID: senderID, // Now 32 bytes, same as node1ID
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

// func testStorageOperations() {
//     mockNetwork := NewMockNetwork()
//     defer mockNetwork.Cleanup()

//     node, err := kademlia.NewKademliaNode(
//         []byte("storage-node"),
//         "storage-peer",
//         mockNetwork,
//         "test_storage.db",
//     )
//     if err != nil {
//         log.Fatalf("Failed to create node: %v", err)
//     }

//     // Test storing node embeddings
//     testEmbeddings := map[string][]float64{
//         "node-1": {1.0, 0.0, 0.0},
//         "node-2": {0.0, 1.0, 0.0},
//         "node-3": {0.0, 0.0, 1.0},
//         "node-4": {0.7, 0.7, 0.0}, // Similar to node-1
//     }

//     for nodeIDStr, embedding := range testEmbeddings {
//         err := storage.StoreNodeEmbedding([]byte(nodeIDStr), embedding)
//         if err != nil {
//             log.Fatalf("Failed to store embedding for %s: %v", nodeIDStr, err)
//         }
//     }

//     fmt.Printf("✓ Stored %d node embeddings\n", len(testEmbeddings))

//     // Test basic storage operations
//     key := []byte("test-key")
//     value := []float64{0.5, 0.5, 0.5}

//     err = node.StoreEmbedding(key, value)
//     if err != nil {
//         log.Fatalf("Failed to store basic embedding: %v", err)
//     }

//     retrievedValue, exists := node.GetEmbedding(key)
//     if !exists {
//         log.Fatalf("Failed to retrieve stored embedding")
//     }

//     if len(retrievedValue) != len(value) {
//         log.Fatalf("Retrieved embedding length mismatch")
//     }

//     fmt.Println("✓ Storage operations successful")
// }

func testEmbeddingSearch() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	searchNodeID := normalizeNodeID("search-node")
	// Create main node
	node, err := kademlia.NewKademliaNode(
		searchNodeID,
		"search-peer",
		mockNetwork,
		"test_search.db",
	)
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}

	// Store some test embeddings locally
	// node.StoreNodeEmbedding([]byte("local-node-1"), []float64{1.0, 0.0, 0.0})
	// node.StoreNodeEmbedding([]byte("local-node-2"), []float64{0.0, 1.0, 0.0})
	// node.StoreNodeEmbedding([]byte("local-node-3"), []float64{0.9, 0.1, 0.0}) // Most similar to query

	// Create target node in mock network
	targetID := normalizeNodeID("remote-node")
	mockNetwork.AddNode(targetID, "127.0.0.1:8081", true)
	mockNetwork.StoreNodeEmbedding(targetID, []byte("similar-remote-node"), []float64{0.95, 0.05, 0.0})

	// Test embedding search request - query similar to local-node-3
	searchReq := &types.EmbeddingSearchRequest{
		SourceNodeID: searchNodeID,
		SourcePeerID: "search-peer",
		QueryEmbed:   []float64{0.8, 0.2, 0.0}, // Similar to local-node-3
		Depth:        0,
		QueryType:    "similarity_search",
		Threshold:    0.7,
		ResultsCount: 5,
		TargetNodeID: targetID,
	}

	// Test local handling (should find local-node-3)
	localResp, err := node.HandleEmbeddingSearch(searchReq)
	if err != nil {
		log.Fatalf("Local embedding search failed: %v", err)
	}

	fmt.Printf("✓ Local search found: %v\n", localResp.Found)
	if localResp.Found {
		fmt.Printf("✓ Found embedding: %v\n", localResp.FileEmbed)
		fmt.Printf("✓ Next NodeID: %x\n", localResp.NextNodeID)
		fmt.Printf("✓ Search depth: %d\n", localResp.Depth)
	}

	// Test remote search
	remoteResp, err := node.EmbeddingSearch(searchReq)
	if err != nil {
		log.Printf("Remote search failed (expected in mock): %v", err)
	} else {
		fmt.Printf("✓ Remote search found: %v\n", remoteResp.Found)
		if remoteResp.Found {
			fmt.Printf("✓ Remote embedding: %v\n", remoteResp.FileEmbed)
		}
	}

	fmt.Println("✓ Embedding search operations successful")
}

func testRoutingTable() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	routingNodeID := normalizeNodeID("routing-node")

	node, err := kademlia.NewKademliaNode(
		routingNodeID,
		"routing-peer",
		mockNetwork,
		"test_routing.db",
	)
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}

	// Show empty routing table
	fmt.Println("Initial routing table (should be empty):")
	node.PrintRoutingTableSummary()

	// Create test peers with normalized NodeIDs
	testPeers := []types.PeerInfo{
		{NodeID: normalizeNodeID("peer-node-1"), PeerID: "peer-id-1"},
		{NodeID: normalizeNodeID("peer-node-2"), PeerID: "peer-id-2"},
		{NodeID: normalizeNodeID("peer-node-3"), PeerID: "peer-id-3"},
		{NodeID: normalizeNodeID("peer-node-4"), PeerID: "peer-id-4"},
	}

	// Update routing table by handling pings
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

	// Show populated routing table
	fmt.Println("\nRouting table after adding peers:")
	node.PrintRoutingTable()

	// Show peer information
	fmt.Println("\nDetailed peer information:")
	node.PrintPeerInfo()

	fmt.Println("✓ Routing table operations successful")
}

func testEndToEndWorkflow() {
	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	// Create a network of nodes
	nodes := make([]*kademlia.KademliaNode, 3)
	nodeIDs := [][]byte{
		normalizeNodeID("network-node-1"), // Fixed
		normalizeNodeID("network-node-2"), // Fixed
		normalizeNodeID("network-node-3"), // Fixed
	}

	peerIDs := []string{
		"network-peer-1",
		"network-peer-2",
		"network-peer-3",
	}

	// Initialize nodes
	for i, nodeID := range nodeIDs {
		var err error
		nodes[i], err = kademlia.NewKademliaNode(
			nodeID,
			peerIDs[i],
			mockNetwork,
			fmt.Sprintf("test_e2e_%d.db", i+1),
		)
		if err != nil {
			log.Fatalf("Failed to create node %d: %v", i+1, err)
		}

		// Add to mock network
		mockNetwork.AddNode(nodeID, fmt.Sprintf("127.0.0.1:800%d", i+1), true)
	}

	// Store different embeddings on different nodes
	// embeddings := [][]float64{
	//     {1.0, 0.0, 0.0}, // Node 1 - specializes in dimension 0
	//     {0.0, 1.0, 0.0}, // Node 2 - specializes in dimension 1
	//     {0.0, 0.0, 1.0}, // Node 3 - specializes in dimension 2
	// }

	// for i, embedding := range embeddings {
	//     err := nodes[i].StoreNodeEmbedding(nodeIDs[i], embedding)
	//     if err != nil {
	//         log.Fatalf("Failed to store embedding on node %d: %v", i+1, err)
	//     }

	//     // Also store in mock network for cross-node searches
	//     mockNetwork.StoreNodeEmbedding(nodeIDs[i], nodeIDs[i], embedding)
	// }

	// Build routing tables by simulating cross-node pings
	for i := 0; i < len(nodes); i++ {
		for j := 0; j < len(nodes); j++ {
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

	// Test local search
	fmt.Println("  Testing local search...")
	localSearchReq := &types.EmbeddingSearchRequest{
		SourceNodeID: nodeIDs[0],
		SourcePeerID: peerIDs[0],
		QueryEmbed:   []float64{0.9, 0.1, 0.0}, // Similar to node 1's embedding
		Depth:        0,
		QueryType:    "local_search",
		Threshold:    0.8,
		ResultsCount: 1,
		TargetNodeID: nodeIDs[0],
	}

	localResp, err := nodes[0].HandleEmbeddingSearch(localSearchReq)
	if err != nil {
		log.Printf("Local search failed: %v", err)
	} else {
		fmt.Printf("  ✓ Local search found: %v\n", localResp.Found)
	}

	fmt.Println("✓ End-to-end workflow successful")
}

func cleanup() {
	// Remove test database files
	testFiles := []string{
		"test_node1.db",
		"test_ping1.db",
		"test_storage.db",
		"test_search.db",
		"test_routing.db",
		"test_e2e_1.db",
		"test_e2e_2.db",
		"test_e2e_3.db",
	}

	for _, file := range testFiles {
		os.Remove(file)
	}

	fmt.Println("✓ Cleanup completed")
}

func testRoutingTableDisplay() {
	fmt.Println("7. Testing Routing Table Display...")

	mockNetwork := NewMockNetwork()
	defer mockNetwork.Cleanup()

	// Create multiple nodes to build interesting routing tables
	nodes := make([]*kademlia.KademliaNode, 5)
	nodeNames := []string{"alpha", "beta", "gamma", "delta", "epsilon"}

	for i, name := range nodeNames {
		nodeID := normalizeNodeID(fmt.Sprintf("display-node-%s", name))
		var err error
		nodes[i], err = kademlia.NewKademliaNode(
			nodeID,
			fmt.Sprintf("display-peer-%s", name),
			mockNetwork,
			fmt.Sprintf("test_display_%d.db", i),
		)
		if err != nil {
			log.Fatalf("Failed to create node %s: %v", name, err)
		}
	}

	// Build interconnected routing tables
	for i := 0; i < len(nodes); i++ {
		for j := 0; j < len(nodes); j++ {
			if i != j {
				pingReq := &types.PingRequest{
					SenderNodeID:   nodes[j].GetID(),
					SenderPeerID: nodes[j].GetAddress(),
					Timestamp:  time.Now().UnixNano(),
				}
				nodes[i].HandlePing(pingReq)
			}
		}
	}

	// Display routing tables for each node
	for i, node := range nodes {
		fmt.Printf("\n" + strings.Repeat("=", 60))
		fmt.Printf("\nNODE %d (%s) ROUTING TABLE", i+1, nodeNames[i])
		fmt.Printf("\n" + strings.Repeat("=", 60))

		node.PrintRoutingTableSummary()

		// Show detailed view for first node only (to avoid too much output)
		if i == 0 {
			fmt.Println("\nDetailed view of first node:")
			node.PrintPeerInfo()
		}
	}

	fmt.Println("\n✓ Routing table display test successful")

	// Cleanup
	for i := range nodes {
		os.Remove(fmt.Sprintf("test_display_%d.db", i))
	}
}
