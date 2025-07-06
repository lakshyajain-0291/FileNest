package main

import (
	"encoding/json"
	"general-peer/pkg/helpers"
	"general-peer/pkg/models"
	"general-peer/pkg/testing"
	"log"
	"time"
)

func main() {
	// Sample query message
	queryMsg := models.Message{
		Type: "query",
		QueryEmbed: testing.GenerateFakeVector(128),
	}

	// Sample peer message
	peerMsg := models.Message{
		Type: "peer",
		QueryEmbed: testing.GenerateFakeVector(128),
		Depth: 3,
		CurrentPeerID: 42,
		IsProcessed: true,
		FileMetadata: models.FileMetadata{
			Name: "example.txt",
			CreatedAt: "2025-07-01T10:00:00Z",
			UpdatedAt: "2025-07-05T12:00:00Z",
		},
	}

	marshalPeerMsg, err := json.Marshal(peerMsg)
	if(err != nil){
		log.Printf("error during marshalling of message: %v", err.Error())
	}

	marshalQueryMsg, err := json.Marshal(queryMsg)
	if(err != nil){
		log.Printf("error during marshalling of message: %v", err.Error())
	}

	generalPeer := &models.Peer{ID: 123, IP: "127.0.0.1", Port: 8080}
	conn, addr := helpers.InitPeer(generalPeer)
	somePeer := &models.Peer{ID: 345, IP: "127.0.0.1", Port: 8081}
	conn_, addr_ := helpers.InitPeer(somePeer)

	msgChan := make(chan models.Message)
	go helpers.ListenForMessage(conn, &msgChan)
	time.Sleep(time.Second*1)
	
	log.Printf("Sending Peer message")
	conn.WriteToUDP(marshalPeerMsg, addr)
	time.Sleep(time.Second*1)
	
	log.Printf("Sending Query message")
	conn.WriteToUDP(marshalQueryMsg, addr)

	err = helpers.ForwardQuery(123,peerMsg.QueryEmbed, addr_, conn_)
	if(err != nil){
		log.Println(err.Error())
	}
	for {
		log.Printf("Recieved Message: %+v", <- msgChan)
	}
}


