package main

import (
	"backend/models"
	"backend/functions"
	"encoding/json"
	"fmt"
	"log"
)
//Only for testing 
func main() {
	// Sample user message (correct 128-dim query_embed)
	userMsg := models.Message{
		Source:     "user",
		QueryEmbed: generateFakeVector(128),
	}

	// Sample peer message (correct 128-dim embed + metadata)
	peerMsg := models.Message{
		Source:      "peer",
		Embed:       generateFakeVector(128),
		Depth:       3,
		PeerID:      42,
		IsProcessed: true,
		FileMetadata: models.FileMetadata{
			Name:      "example.txt",
			CreatedAt: "2025-07-01T10:00:00Z",
			UpdatedAt: "2025-07-05T12:00:00Z",
		},
	}

	// Serialize and test both
	test(userMsg)
	test(peerMsg)
}

// Test helper
func test(msg models.Message) {

	jsonData, err := json.Marshal(msg)
	if err != nil {
		log.Fatal("Failed to marshal JSON:", err)
	}

	parsed, err := functions.ParseMessage(jsonData)
	if err != nil {
		log.Println("Parse error:", err)
	} else {
		fmt.Printf("Successfully parsed: %+v\n", parsed)
	}
}

// Generate dummy float64 vector
func generateFakeVector(n int) []float64 {
	vec := make([]float64, n)
	for i := 0; i < n; i++ {
		vec[i] = float64(i) / float64(n)
	}
	return vec
}
