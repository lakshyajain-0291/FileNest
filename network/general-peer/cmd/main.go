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

	// // Start TCP listener for incoming messages
	// listener := helpers.InitTCPListener("127.0.0.1:8080")
	msgChan := make(chan models.Message)
	// go helpers.ListenForMessage(listener, msgChan)

	// Create WebRTC host
	webRTCHost, err := helpers.CreateWebRTCHost(12345)
	if err != nil {
		log.Fatalf("Failed to create WebRTC host: %v", err)
	}
	defer webRTCHost.Close()
	log.Printf("WebRTC Host ID: %s", webRTCHost.ID())
	// log.Printf("WebRTC Addresses: %v", webRTCHost.Addrs())

	webRTCHost2, err := helpers.CreateWebRTCHost(12346)
	if err != nil {
		log.Fatalf("Failed to create WebRTC host: %v", err)
	}
	defer webRTCHost2.Close()
	log.Printf("WebRTC Host ID: %s", webRTCHost2.ID())
	// log.Printf("WebRTC Addresses: %v", webRTCHost2.Addrs())

	// Set stream handler for WebRTC

	// webRTCHost2.SetStreamHandler("/general-peer/query", func(s network.Stream) {
	// 	var msg models.Message
	// 	if err := json.NewDecoder(s).Decode(&msg); err != nil {
	// 		log.Printf("WebRTC decode error (host): %v", err)
	// 		s.Reset()
	// 		return
	// 	}
	// 	s.Close()
	// 	msgChan <- msg
	// 	log.Printf("Received WebRTC message (host): %+v", msg)
	// })

	time.Sleep(time.Second)
	// Send Peer message over WebRTC (to ourselves)
	log.Println("Sending Peer message via WebRTC...")
	selfAddr := helpers.GetHostAddress(webRTCHost2)
	err = helpers.SendWebRTCMessage(webRTCHost, selfAddr, peerMsg, msgChan, "general-peer/peer")
	if err != nil {
		log.Printf("Error sending WebRTC message: %v", err)
	}

	time.Sleep(time.Second)

	log.Println("Sending Query message via WebRTC...")
	err = helpers.SendWebRTCMessage(webRTCHost, selfAddr, queryMsg, msgChan, "general-peer/query")
	if err != nil {
		log.Printf("Error sending WebRTC message: %v", err)
	}

	for {
		msg := <-msgChan
		log.Printf("Received Message: %+v", msg)
	}
}
