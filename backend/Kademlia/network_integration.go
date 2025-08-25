package main

import (
	"crypto/rand"
	"errors"
	"fmt"
	"kademlia/pkg/helpers"
	"kademlia/pkg/identity"
	"kademlia/pkg/kademlia"
	"kademlia/pkg/types"
	"log"
	"math/big"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

// This file contains the network integration logic for the Kademlia DHT implementation.

// NetworkIntegrationService handles network layer integration with Kademlia DHT
type NetworkIntegrationService struct {
	kademliaNode       *kademlia.KademliaNode
	embeddingProcessor *helpers.EmbeddingProcessor
	hostPeerID         peer.ID
	currentDepth       int
}

// NewNetworkIntegrationService creates a new network integration service
func NewNetworkIntegrationService(node *kademlia.KademliaNode, processor *helpers.EmbeddingProcessor, depth int) *NetworkIntegrationService {
	return &NetworkIntegrationService{
		kademliaNode:       node,
		embeddingProcessor: processor,
		hostPeerID:         peer.ID(node.GetAddress()),
		currentDepth:       depth,
	}
}

// ProcessEmbeddingRequest handles all incoming embedding requests and routes/answers them appropriately.
func (nis *NetworkIntegrationService) ProcessEmbeddingRequest(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Processing embedding request: type=%s, target=%x, source=%s, depth=%d",
		request.QueryType, request.TargetNodeID[:8], request.SourcePeerID, request.Depth)

	// Determine if this node is the target
	myNodeID := nis.kademliaNode.GetID()

	// Compare node IDs properly (byte slice comparison)
	if bytesEqual(request.TargetNodeID, myNodeID) {
		log.Printf("Target node matches current peer - processing locally")
		return nis.processLocalEmbeddingTarget(request)
	} else {
		log.Printf("Target node doesn't match - finding closest peer for forwarding")
		return nis.forwardEmbeddingToClosestPeer(request)
	}
}

// processLocalEmbeddingTarget handles when current node is the target for embedding operations
func (nis *NetworkIntegrationService) processLocalEmbeddingTarget(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Processing embedding request locally for target node %x, type: %s", request.TargetNodeID[:8], request.QueryType)

	// Handle different types of embedding requests
	switch request.QueryType {
	case "search":
		return nis.handleEmbeddingSearch(request)
	case "store":
		return nis.handleEmbeddingStore(request)
	default:
		return nis.handleGenericEmbedding(request)
	}
}

// handleEmbeddingSearch processes embedding search requests
func (nis *NetworkIntegrationService) handleEmbeddingSearch(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Handling embedding search at depth %d for target %x", nis.currentDepth, request.TargetNodeID[:8])

	// Use the Kademlia node's built-in embedding search functionality
	response, err := nis.kademliaNode.HandleEmbeddingSearch(request)
	if err != nil {
		log.Printf("Embedding search processing failed: %v", err)
		return &types.EmbeddingSearchResponse{
			QueryType:    "search_error",
			QueryEmbed:   request.QueryEmbed,
			Depth:        nis.currentDepth,
			SourceNodeID: nis.kademliaNode.GetID(),
			SourcePeerID: nis.kademliaNode.GetAddress(),
			Found:        false,
			NextNodeID:   nil,
			FileEmbed:    nil,
		}, err
	}
	return response, nil
}

// handleEmbeddingStore processes embedding storage requests
func (nis *NetworkIntegrationService) handleEmbeddingStore(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Handling embedding store at depth %d for target %x", nis.currentDepth, request.TargetNodeID[:8])

	// Store the embedding using the Kademlia node's storage
	err := nis.kademliaNode.StoreNodeEmbedding(request.TargetNodeID, request.QueryEmbed)
	if err != nil {
		log.Printf("Embedding store processing failed: %v", err)
		return &types.EmbeddingSearchResponse{
			QueryType:    "store_error",
			QueryEmbed:   request.QueryEmbed,
			Depth:        nis.currentDepth,
			SourceNodeID: nis.kademliaNode.GetID(),
			SourcePeerID: nis.kademliaNode.GetAddress(),
			Found:        false,
			NextNodeID:   nil,
			FileEmbed:    nil,
		}, err
	}

	return &types.EmbeddingSearchResponse{
		QueryType:    "store_response",
		QueryEmbed:   request.QueryEmbed,
		Depth:        nis.currentDepth,
		SourceNodeID: nis.kademliaNode.GetID(),
		SourcePeerID: nis.kademliaNode.GetAddress(),
		Found:        true,
		NextNodeID:   nil,
		FileEmbed:    request.QueryEmbed,
	}, nil
}

// handleGenericEmbedding processes other types of embedding requests
func (nis *NetworkIntegrationService) handleGenericEmbedding(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Handling generic embedding request of type: %s", request.QueryType)

	// Use embedding processor if available
	if nis.embeddingProcessor != nil {
		embedding, err := nis.embeddingProcessor.ProcessEmbedding(request.QueryEmbed)
		if err != nil {
			log.Printf("Generic embedding processing failed: %v", err)
		} else {
			return &types.EmbeddingSearchResponse{
				QueryType:    "processed",
				QueryEmbed:   embedding,
				Depth:        nis.currentDepth,
				SourceNodeID: nis.kademliaNode.GetID(),
				SourcePeerID: nis.kademliaNode.GetAddress(),
				Found:        true,
				NextNodeID:   nil,
				FileEmbed:    embedding,
			}, nil
		}
	}

	// Default response for unknown types
	return &types.EmbeddingSearchResponse{
		QueryType:    "processed",
		QueryEmbed:   request.QueryEmbed,
		Depth:        nis.currentDepth,
		SourceNodeID: nis.kademliaNode.GetID(),
		SourcePeerID: nis.kademliaNode.GetAddress(),
		Found:        false,
		NextNodeID:   nis.findNextNodeForSearch(request),
		FileEmbed:    nil,
	}, nil
}

// forwardEmbeddingToClosestPeer finds the closest peer to target node id and forwards the embedding message
func (nis *NetworkIntegrationService) forwardEmbeddingToClosestPeer(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Finding closest peers to target %x for forwarding", request.TargetNodeID[:8])

	// Use Kademlia routing table to find closest peers
	rt := nis.kademliaNode.RoutingTable()
	closestPeers := rt.FindClosest(request.TargetNodeID, rt.K)

	if len(closestPeers) == 0 {
		log.Printf("No peers found for forwarding to target %x", request.TargetNodeID[:8])
		return &types.EmbeddingSearchResponse{
			QueryType:    "no_peers",
			QueryEmbed:   request.QueryEmbed,
			Depth:        request.Depth,
			SourceNodeID: nis.kademliaNode.GetID(),
			SourcePeerID: nis.kademliaNode.GetAddress(),
			Found:        false,
			NextNodeID:   nil,
			FileEmbed:    nil,
		}, fmt.Errorf("no peers available for forwarding")
	}

	// Use the closest peer for forwarding
	closestPeer := closestPeers[0]
	log.Printf("Forwarding embedding request to closest peer: %s (node ID: %x)", closestPeer.PeerID, closestPeer.NodeID[:8])

	return &types.EmbeddingSearchResponse{
		QueryType:    "forward",
		QueryEmbed:   request.QueryEmbed,
		Depth:        request.Depth,
		SourceNodeID: nis.kademliaNode.GetID(),
		SourcePeerID: nis.kademliaNode.GetAddress(),
		Found:        false,
		NextNodeID:   closestPeer.NodeID,
		FileEmbed:    nil,
	}, nil
}

// findNextNodeForSearch determines the next node for embedding search operations
func (nis *NetworkIntegrationService) findNextNodeForSearch(request *types.EmbeddingSearchRequest) []byte {
	// If we're at D4 (depth 4), no next node
	if nis.currentDepth >= 4 {
		return nil
	}

	// Use routing table to find a suitable next node
	rt := nis.kademliaNode.RoutingTable()
	closestPeers := rt.FindClosest(request.TargetNodeID, 1)

	if len(closestPeers) > 0 {
		return closestPeers[0].NodeID
	}

	return nil
}

// ProcessBatchEmbeddingRequests processes multiple embedding requests
func (nis *NetworkIntegrationService) ProcessBatchEmbeddingRequests(requests []*types.EmbeddingSearchRequest) ([]*types.EmbeddingSearchResponse, error) {
	responses := make([]*types.EmbeddingSearchResponse, 0, len(requests))

	for i, request := range requests {
		log.Printf("Processing batch request %d/%d", i+1, len(requests))

		response, err := nis.ProcessEmbeddingRequest(request)
		if err != nil {
			log.Printf("Batch request %d failed: %v", i+1, err)
			// Create error response
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

// Helper function to compare byte slices
func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Helper functions from helpers package (kept here for compatibility)

// ParseBootstrapAddr parses and validates a bootstrap address string.
func ParseBootstrapAddr(addr string) (peer.AddrInfo, error) {
	maddr, err := peer.AddrInfoFromString(addr)
	if err != nil {
		return peer.AddrInfo{}, errors.New("invalid bootstrap node multiaddr")
	}
	return *maddr, nil
}

// XORDistance computes the XOR distance between two node IDs.
func XORDistance(a, b []byte) *big.Int {
	if len(a) != len(b) {
		panic("IDs must be the same length")
	}
	dist := make([]byte, len(a))
	for i := range a {
		dist[i] = a[i] ^ b[i]
	}
	return new(big.Int).SetBytes(dist)
}

// BucketIndex returns the index of the bucket for a given node ID.
func BucketIndex(selfID, otherID []byte) int {
	if len(selfID) != len(otherID) {
		panic("IDs must be the same length")
	}
	for byteIndex := range selfID {
		xorByte := selfID[byteIndex] ^ otherID[byteIndex]
		if xorByte != 0 {
			for bitPos := range 8 {
				if (xorByte & (0x80 >> bitPos)) != 0 {
					return (len(selfID)-byteIndex-1)*8 + (7 - bitPos)
				}
			}
		}
	}
	return -1 // identical IDs → no bucket
}

// RandomNodeID generates a random node ID of the correct length.
func RandomNodeID() []byte {
	id := make([]byte, identity.NodeIDBytes)
	if _, err := rand.Read(id); err != nil {
		log.Fatalf("failed to generate random NodeID: %v", err)
	}
	return id
}

// Example usage function demonstrating how network layer would use this integration
func ExampleEmbeddingNetworkIntegration(node *kademlia.KademliaNode, processor *helpers.EmbeddingProcessor, depth int) {
	// Create network integration service
	nis := NewNetworkIntegrationService(node, processor, depth)

	// Example embedding search request
	searchRequest := &types.EmbeddingSearchRequest{
		QueryType:    "search",
		QueryEmbed:   []float64{0.1, 0.2, 0.3, 0.4, 0.5},
		Depth:        0,
		SourceNodeID: node.GetID(),
		SourcePeerID: node.GetAddress(),
		TargetNodeID: RandomNodeID(), // Generate random target for demo
		Threshold:    0.8,
		ResultsCount: 10,
	}

	// Validate the request
	if err := nis.ValidateEmbeddingRequest(searchRequest); err != nil {
		log.Printf("Request validation failed: %v", err)
		return
	}

	// Process the embedding search request
	response, err := nis.ProcessEmbeddingRequest(searchRequest)
	if err != nil {
		log.Printf("Embedding search failed: %v", err)
	} else {
		log.Printf("Embedding search response: type=%s, found=%t",
			response.QueryType, response.Found)
	}

	// Example embedding store request
	storeRequest := &types.EmbeddingSearchRequest{
		QueryType:    "store",
		QueryEmbed:   []float64{0.2, 0.3, 0.4, 0.5, 0.6},
		Depth:        1,
		SourceNodeID: node.GetID(),
		SourcePeerID: node.GetAddress(),
		TargetNodeID: node.GetID(), // Store locally for demo
		Threshold:    0.7,
		ResultsCount: 5,
	}

	// Process the embedding store request
	response, err = nis.ProcessEmbeddingRequest(storeRequest)
	if err != nil {
		log.Printf("Embedding store failed: %v", err)
	} else {
		log.Printf("Embedding store response: type=%s, found=%t",
			response.QueryType, response.Found)
	}

	// Show network integration stats
	stats := nis.GetNetworkIntegrationStats()
	log.Printf("Network integration stats: %+v", stats)
}
