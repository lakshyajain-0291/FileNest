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
	sendReq := flag.Bool("sendreq", false, "Whether to send request to target peer")
	pid := flag.String("pid", "", "Target peer ID to send request to (used only if sendreq=true)")
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

	// Start Depth Peer
	p, err := peer.NewDepthPeer(relayAddrs)
	if err != nil {
		log.Printf("Error on NewDepthPeer: %v\n", err.Error())
	}

	ctx := context.Background()
	peer.Start(p, ctx)

	// Conditionally send request if flag is set
	if *sendReq {
		if *pid == "" {
			log.Fatal("You must provide a -pid value when using -sendreq")
		}
		
		// stores extra data like routing
		params := models.PingRequest{
			Type: "GET",
			Route: "ping",
			ReceiverPeerID: *pid,
			Timestamp: time.Now().Unix(),
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

	// Block main goroutine
	select {}
}
