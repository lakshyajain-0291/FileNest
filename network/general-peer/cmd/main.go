// package main

// import (
// 	"encoding/json"
// 	"general-peer/pkg/helpers"
// 	"general-peer/pkg/models"
// 	"general-peer/pkg/testing"
// 	"log"
// 	"time"

// 	"github.com/libp2p/go-libp2p/core/network"
// )

// func main() {
// 	queryMsg := models.Message{
// 		Type:       "query",
// 		QueryEmbed: testing.GenerateFakeVector(3),
// 	}

// 	peerMsg := models.Message{
// 		Type:          "peer",
// 		QueryEmbed:    testing.GenerateFakeVector(3),
// 		Depth:         3,
// 		CurrentPeerID: 42,
// 		IsProcessed:   true,
// 		FileMetadata: models.FileMetadata{
// 			Name:      "example.txt",
// 			CreatedAt: "2025-07-01T10:00:00Z",
// 			UpdatedAt: "2025-07-05T12:00:00Z",
// 		},
// 	}

// 	msgChan := make(chan models.Message)

// 	// Create WebRTC host
// 	senderHost, err := helpers.CreateWebRTCHost(12345)
// 	if err != nil {
// 		log.Fatalf("Failed to create WebRTC host: %v", err)
// 	}
// 	defer senderHost.Close()
// 	log.Printf("WebRTC Host ID: %s", senderHost.ID())

// 	recieverHost, err := helpers.CreateWebRTCHost(12346)
// 	if err != nil {
// 		log.Fatalf("Failed to create WebRTC host: %v", err)
// 	}
// 	defer recieverHost.Close()
// 	log.Printf("WebRTC Host ID: %s", recieverHost.ID())

// 	recieverHost.SetStreamHandler("/general-peer/query", func(s network.Stream) {
//     var msg models.Message
//     if err := json.NewDecoder(s).Decode(&msg); err != nil {
//         log.Printf("WebRTC decode error (host): %v", err)
//         s.Reset()
//         return
//     }
//     s.Close()
//     msgChan <- msg
// })

// recieverHost.SetStreamHandler("/general-peer/peer", func(s network.Stream) {
// 	var msg models.Message
// 	if err := json.NewDecoder(s).Decode(&msg); err != nil {
// 		log.Printf("WebRTC decode error (host): %v", err)
// 		s.Reset()
// 		return
// 	}
// 	s.Close()
// 	msgChan <- msg
// })

// 	time.Sleep(time.Second)
// 	// Send Peer message over WebRTC (to ourselves)
// 	log.Println("Sending Peer message via WebRTC...")
// 	recieverAddr := helpers.GetHostAddress(recieverHost)
// 	err = helpers.SendWebRTCMessage(senderHost, recieverAddr, peerMsg, "/general-peer/peer")
// 	if err != nil {
// 		log.Printf("Error sending WebRTC message: %v", err)
// 	}

// 	time.Sleep(time.Second)

// 	log.Println("Sending Query message via WebRTC...")
// 	err = helpers.SendWebRTCMessage(senderHost, recieverAddr, queryMsg, "/general-peer/query")
// 	if err != nil {
// 		log.Printf("Error sending WebRTC message: %v", err)
// 	}

//		for {
//			msg := <-msgChan
//			log.Printf("Received Message: %+v", msg)
//		}
//	}
package main

import (
	"encoding/json"
	"general-peer/pkg/helpers"
	"general-peer/pkg/ml_client"
	"general-peer/pkg/models"
	"general-peer/pkg/testing"
	"log"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
)

func main() {
	// Initialize ML client
	mlClient, err := ml_client.NewMLClient("tcp://localhost:5555")
	if err != nil {
		log.Fatalf("Failed to create ML client: %v", err)
	}
	defer mlClient.Close()

	// Example: Generate embedding
	embedding, err := mlClient.GenerateEmbedding("example text to embed")
	if err != nil {
		log.Printf("Error generating embedding: %v", err)
	} else {
		log.Printf("Generated embedding with length: %d", len(embedding))
	}

	// Example: Generate clusters
	clusters, err := mlClient.GenerateClusters("path/to/text/files")
	if err != nil {
		log.Printf("Error generating clusters: %v", err)
	} else {
		log.Printf("Generated %d clusters", len(clusters.Clusters))
	}

	// Rest of your existing WebRTC code...
	queryMsg := models.Message{
		Type:       "query",
		QueryEmbed: testing.GenerateFakeVector(3),
	}

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

	msgChan := make(chan models.Message)

	// Create WebRTC host
	senderHost, err := helpers.CreateWebRTCHost(12345)
	if err != nil {
		log.Fatalf("Failed to create WebRTC host: %v", err)
	}
	defer senderHost.Close()
	log.Printf("WebRTC Host ID: %s", senderHost.ID())

	recieverHost, err := helpers.CreateWebRTCHost(12346)
	if err != nil {
		log.Fatalf("Failed to create WebRTC host: %v", err)
	}
	defer recieverHost.Close()
	log.Printf("WebRTC Host ID: %s", recieverHost.ID())

	recieverHost.SetStreamHandler("/general-peer/query", func(s network.Stream) {
		var msg models.Message
		if err := json.NewDecoder(s).Decode(&msg); err != nil {
			log.Printf("WebRTC decode error (host): %v", err)
			s.Reset()
			return
		}
		s.Close()
		msgChan <- msg
	})

	recieverHost.SetStreamHandler("/general-peer/peer", func(s network.Stream) {
		var msg models.Message
		if err := json.NewDecoder(s).Decode(&msg); err != nil {
			log.Printf("WebRTC decode error (host): %v", err)
			s.Reset()
			return
		}
		s.Close()
		msgChan <- msg
	})

	time.Sleep(time.Second)
	// Send Peer message over WebRTC (to ourselves)
	log.Println("Sending Peer message via WebRTC...")
	recieverAddr := helpers.GetHostAddress(recieverHost)
	err = helpers.SendWebRTCMessage(senderHost, recieverAddr, peerMsg, "/general-peer/peer")
	if err != nil {
		log.Printf("Error sending WebRTC message: %v", err)
	}

	time.Sleep(time.Second)

	log.Println("Sending Query message via WebRTC...")
	err = helpers.SendWebRTCMessage(senderHost, recieverAddr, queryMsg, "/general-peer/query")
	if err != nil {
		log.Printf("Error sending WebRTC message: %v", err)
	}

	for {
		msg := <-msgChan
		log.Printf("Received Message: %+v", msg)
	}
}
