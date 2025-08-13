package main

import (
	"general-peer/pkg/models"
	"general-peer/pkg/testing"
	"general-peer/pkg/ws"
	"log"
	"time"
)

func main() {
	// --- Network Transport ---
	msgChan := make(chan models.Message)
	mlChan := make(chan models.ClusterWrapper)

	peerTransport := ws.NewWebSocketTransport(":8080")
	testTransport := ws.NewWebSocketTransport(":8081")
	mlTransport := ws.NewWebSocketTransport(":8082")
	defer func () {
		mlTransport.Close()
		peerTransport.Close()
		testTransport.Close()
	}()
	
	go func() {
		if err := peerTransport.StartPeerReceiver(msgChan); err != nil {
			log.Printf("Peer Receiver error: %v", err)
		}
	}()

	go func(){
		if err := mlTransport.StartMLReceiver(mlChan); err != nil {
			log.Printf("ML Receiver error: %v", err)
		}
	}()

	peerMsg := models.Message{
		Type:          "peer",
		QueryEmbed:    testing.GenerateFakeVector(3),
		Depth:         3,
		CurrentPeerID: 42,
		IsProcessed:   true,
		FileMetadata: models.FileMetadata{
			Name:      "example.txt",
			CreatedAt: "2025-07-01T10:00:00Z",
			UpdatedAt: "2025-07-05T12:00:00Z",
		},
	}
	time.Sleep((time.Second*2))
	if err := testTransport.SendMessage("ws://127.0.0.1:8080/peer", peerMsg); err != nil {
		log.Printf("[NET] Send error: %v", err)
	}

	// ML Receive Loop
	go func() {
		log.Println("[NET] Listening for ML messages...")
		for {
			msg := <-mlChan
			log.Printf("[NET] Received ML message: %+v", msg)
		}
	}()

	// Network recieve loop
	go func() {
		log.Println("[NET] Listening for peer messages...")
		for {
			msg := <-msgChan
			log.Printf("[NET] Received Peer message: %+v", msg)
		}
	}()

	select {}
}
