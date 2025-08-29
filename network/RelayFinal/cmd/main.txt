package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	genmodels "network/pkg/generalpeer/models"
	"network/pkg/generalpeer/ws"
	"network/pkg/relay/helpers"
	"network/pkg/relay/models"
	"network/pkg/relay/peer"
	"time"
)

func main() {
	var mlChan chan genmodels.ClusterWrapper

	// Define flags
	pid := flag.String("pid", "", "Target peer ID to send request to (optional)")
	peerType := flag.String("peertype", "user", "Type of peer to start: 'user' or 'depth'")
	flag.Parse()

	peerTransport := ws.NewWebSocketTransport(":8080")
	mlTransport := ws.NewWebSocketTransport(":8081")
	defer func() {
		mlTransport.Close()
		peerTransport.Close()
	}()

	// Start ML Receiver
	go func() {
		if err := mlTransport.StartMLReceiver(mlChan); err != nil {
			log.Printf("ML Receiver error: %v", err)
		}
	}()

	// Consume ML messages
	go func() {
		log.Println("[NET] Listening for ML messages...")
		for {
			msg := <-mlChan
			log.Printf("[NET] Received ML message: %+v", msg)
		}
	}()

	// Get relay addresses
	relayAddrs, err := helpers.GetRelayAddrFromMongo()
	if err != nil {
		log.Printf("Error during get relay addrs: %v", err.Error())
	}
	log.Printf("relayAddrs in Mongo: %+v\n", relayAddrs)

	// Initialize peer based on flag
	ctx := context.Background()
	p, err := peer.NewPeer(relayAddrs, *peerType)
	if err != nil {
		log.Fatalf("Error creating %s peer: %v", *peerType, err)
	}
	peer.Start(p, ctx)
	log.Printf("Started %s peer successfully", *peerType)

	// If pid is provided, send request
	if *pid != "" {
		params := models.PingRequest{
			Type:           "GET",
			Route:          "ping",
			ReceiverPeerID: *pid,
			Timestamp:      time.Now().Unix(),
		}
		jsonParams, _ := json.Marshal(params)

		resp, err := peer.Send(p, ctx, *pid, jsonParams, nil)
		if err != nil {
			log.Println(err.Error())
		}

		var respDec any
		json.Unmarshal(resp, &respDec)
		log.Printf("Response: %+v", respDec)
	}

	select {}
}
