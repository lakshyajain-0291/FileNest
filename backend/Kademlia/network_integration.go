package main

import (
	"bytes"
	"crypto/rand"
	"errors"
	"fmt"
	"kademlia/pkg/helpers"
	"kademlia/pkg/identity"
	"kademlia/pkg/kademlia"
	"kademlia/pkg/types"
	"log"
	"math"
	"math/big"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

// Local wrapper type to extend helpers.EmbeddingProcessor
// This solves the "cannot define new methods on non-local type" error
type LocalEmbeddingProcessor struct {
	*helpers.EmbeddingProcessor
}

// NewLocalEmbeddingProcessor creates a wrapper around helpers.EmbeddingProcessor
func NewLocalEmbeddingProcessor() *LocalEmbeddingProcessor {
	return &LocalEmbeddingProcessor{
		EmbeddingProcessor: &helpers.EmbeddingProcessor{},
	}
}

// CosineSimilarity method added to local wrapper (fixes the error)
func (ep *LocalEmbeddingProcessor) CosineSimilarity(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("embedding dimensions don't match: %d != %d", len(a), len(b))
	}

	if len(a) == 0 {
		return 0, fmt.Errorf("empty embedding vectors")
	}

	var dotProduct, normA, normB float64

	// Calculate dot product and norms in one pass
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	// Handle zero vectors
	if normA == 0 || normB == 0 {
		return 0.0, nil
	}

	// Calculate cosine similarity
	similarity := dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))

	return similarity, nil
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

	// Get all stored embeddings from the node's database
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

	// Find the most similar embedding using cosine similarity
	var bestMatch *EmbeddingResult
	maxSimilarity := -2.0 // Start with value lower than minimum possible similarity

	for _, stored := range storedEmbeddings {
		similarity, err := nis.embeddingProcessor.CosineSimilarity(request.QueryEmbed, stored.Embedding)
		if err != nil {
			continue // Skip invalid embeddings
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

	// Return response with next node ID and Found = true (as requested)
	return &types.EmbeddingSearchResponse{
		QueryType:    "similarity_match",
		QueryEmbed:   request.QueryEmbed,
		Depth:        request.Depth + 1,
		SourceNodeID: nis.kademliaNode.GetID(),
		SourcePeerID: nis.kademliaNode.GetAddress(),
		Found:        true,             // Found = true as requested
		NextNodeID:   bestMatch.NodeID, // Next node ID from similarity match
		FileEmbed:    bestMatch.Embedding,
	}, nil
}

// routeViaKademlia - When target doesn't match, route via Kademlia DHT
func (nis *NetworkIntegrationService) routeViaKademlia(request *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
	log.Printf("Routing to target node %x via Kademlia", request.TargetNodeID[:8])

	// Use Kademlia routing table to find the closest peer to the target
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

	// Get the closest peer for forwarding
	nextHop := closestPeers[0]

	log.Printf("Routing to next hop: %s (node ID: %x)", nextHop.PeerID, nextHop.NodeID[:8])

	return &types.EmbeddingSearchResponse{
		QueryType:    "routed",
		QueryEmbed:   request.QueryEmbed,
		Depth:        request.Depth,
		SourceNodeID: nis.kademliaNode.GetID(),
		SourcePeerID: nis.kademliaNode.GetAddress(),
		Found:        false,          // Found = false when routing
		NextNodeID:   nextHop.NodeID, // Next hop from Kademlia routing
		FileEmbed:    nil,
	}, nil
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

// Supporting type for embedding results
type EmbeddingResult struct {
	NodeID     []byte    `json:"node_id"`
	Embedding  []float64 `json:"embedding"`
	Similarity float64   `json:"similarity"`
}

// Helper functions from helpers package
func ParseBootstrapAddr(addr string) (peer.AddrInfo, error) {
	maddr, err := peer.AddrInfoFromString(addr)
	if err != nil {
		return peer.AddrInfo{}, errors.New("invalid bootstrap node multiaddr")
	}
	return *maddr, nil
}

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
	return -1
}

func RandomNodeID() []byte {
	id := make([]byte, identity.NodeIDBytes)
	if _, err := rand.Read(id); err != nil {
		log.Fatalf("failed to generate random NodeID: %v", err)
	}
	return id
}

// Example usage function
func ExampleEmbeddingNetworkIntegration(node *kademlia.KademliaNode, depth int) {
	// Create local embedding processor
	processor := NewLocalEmbeddingProcessor()

	// Create network integration service
	nis := NewNetworkIntegrationService(node, processor, depth)

	// Example embedding search request
	searchRequest := &types.EmbeddingSearchRequest{
		QueryType:    "search",
		QueryEmbed:   []float64{0.1, 0.2, 0.3, 0.4, 0.5},
		Depth:        0,
		SourceNodeID: node.GetID(),
		SourcePeerID: node.GetAddress(),
		TargetNodeID: RandomNodeID(),
		Threshold:    0.8,
		ResultsCount: 10,
	}

	// Validate the request
	if err := nis.ValidateEmbeddingRequest(searchRequest); err != nil {
		log.Printf("Request validation failed: %v", err)
		return
	}

	// Process the embedding request - USES NEW ROUTING LOGIC
	response, err := nis.ProcessEmbeddingRequest(searchRequest)
	if err != nil {
		log.Printf("Embedding processing failed: %v", err)
	} else {
		log.Printf("Response: type=%s, found=%t, next_node=%x",
			response.QueryType, response.Found, response.NextNodeID)
	}

	// Show network integration stats
	stats := nis.GetNetworkIntegrationStats()
	log.Printf("Network integration stats: %+v", stats)
}
