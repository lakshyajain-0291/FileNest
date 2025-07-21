package main

import (
	"general-peer/pkg/helpers"
	"general-peer/pkg/models"
	"general-peer/pkg/testing"
	"log"
	"time"
)

func main() {
	queryMsg := models.Message{
		Type:       "query",
		QueryEmbed: testing.GenerateFakeVector(128),
	}

	peerMsg := models.Message{
		Type:          "peer",
		QueryEmbed:    testing.GenerateFakeVector(128),
		Depth:         3,
		CurrentPeerID: 42,
		IsProcessed:   true,
		FileMetadata: models.FileMetadata{
			Name:      "example.txt",
			CreatedAt: "2025-07-01T10:00:00Z",
			UpdatedAt: "2025-07-05T12:00:00Z",
		},
	}

	// Start TCP listener for incoming messages
	listener := helpers.InitTCPListener("127.0.0.1:8080")
	msgChan := make(chan models.Message)
	go helpers.ListenForMessage(listener, msgChan)

	time.Sleep(time.Second)

	// Send Peer message over TCP
	log.Println("Sending Peer message via TCP...")
	err := helpers.SendTCPMessage("127.0.0.1:8081", peerMsg)
	if err != nil {
		log.Printf("Error sending peer message: %v", err)
	}

	time.Sleep(time.Second)

	// Send Query message over TCP
	log.Println("Sending Query message via TCP...")
	err = helpers.SendTCPMessage("127.0.0.1:8081", queryMsg)
	if err != nil {
		log.Printf("Error sending query message: %v", err)
	}

	err = helpers.ForwardQueryTCP(123, peerMsg.QueryEmbed, "127.0.0.1:8081")
	if err != nil {
		log.Printf("Error forwarding query: %v", err)
	}

	for {
		msg := <-msgChan
		log.Printf("Received Message: %+v", msg)
	}
}
