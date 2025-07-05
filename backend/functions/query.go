package functions

import (
	"backend/models"
	"encoding/json"
	"errors"
	"fmt"
)

// ForwardQuery forwards a user query to all Depth 1 peers and aggregates their responses.
// - data: raw JSON bytes received from the client
// - depth1Peers: list of depth 1 peer addresses/IDs
// Returns all results aggregated into a single list.
func ForwardQuery(data []byte, depth1Peers []string) ([]models.SearchResult, error) {
	// Parse and validate the incoming message
	msg, err := ParseMessage(data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse query message: %w", err)
	}

	if msg.Source != "user" {
		return nil, errors.New("only user queries are allowed in ForwardQuery")
	}

	if len(depth1Peers) == 0 {
		return nil, errors.New("no depth 1 peers configured for forwarding")
	}

	var aggregated []models.SearchResult

	for _, peerID := range depth1Peers {
		// Forward to peer (this is mocked here)
		results, err := forwardQueryToPeer(peerID, msg)
		if err != nil {
			fmt.Printf("Warning: peer %s returned error: %v\n", peerID, err)
			continue
		}
		aggregated = append(aggregated, results...)
	}

	return aggregated, nil
}

// forwardQueryToPeer sends the query to a single peer and receives results.
// In production, replace this with actual network code (HTTP/libp2p).
func forwardQueryToPeer(peerID string, query *models.Message) ([]models.SearchResult, error) {
	// Simulate serialization
	payload, err := json.Marshal(query)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal query: %w", err)
	}

	// Simulated peer request/response
	// TODO: Replace this with real network call
	fmt.Printf("Forwarding query to peer %s...\n", peerID)
	mockResponse := []models.SearchResult{
		{
			FileURI:    fmt.Sprintf("file://%s/fake1.txt", peerID),
			OriginPeer: peerID,
			Similarity: 0, // The depth peer will fill in real similarity
		},
	}

	// Simulate success
	_ = payload // suppress unused warning in mock

	return mockResponse, nil
}
