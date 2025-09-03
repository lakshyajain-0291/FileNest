package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	genmodels "final/network/RelayFinal/pkg/generalpeer/models"
	"final/network/RelayFinal/pkg/generalpeer/ws"
	"final/network/RelayFinal/pkg/network/helpers"
	relayhelper "final/network/RelayFinal/pkg/relay/helpers"
	"final/network/RelayFinal/pkg/relay/models"
	"final/network/RelayFinal/pkg/relay/peer"
	"flag"
	"log"
	"time"

	"final/backend/pkg/integration"
	"final/backend/pkg/types"
)

func main() {
	// ---- Flags ----
	sendReq := flag.Bool("sendreq", false, "Whether to send request to target peer")
	pid := flag.String("pid", "", "Target peer ID to send request to (comma sep)")
	nodeidHex := flag.String("nodeid", "", "Node ID for this peer (hex, comma sep)")
	useKademlia := flag.Bool("kademlia", true, "Use Kademlia for embedding search")
	flag.Parse()

	// ---- Decode nodeid ----
	var nodeidBytes []byte
	if *nodeidHex != "" {
		var err error
		nodeidBytes, err = hex.DecodeString(*nodeidHex)
		if err != nil {
			log.Fatalf("invalid bootstrap NodeID: %v", err)
		}
	}

	// ---- Setup ML transport ----
	mlChan := make(chan genmodels.ClusterWrapper, 10)
	mlTransport := ws.NewWebSocketTransport(":8081")
	defer mlTransport.Close()

	go func() {
		if err := mlTransport.StartMLReceiver(mlChan); err != nil {
			log.Printf("ML Receiver error: %v", err)
		}
	}()

	go func() {
		log.Println("[NET] Listening for ML messages...")
		for msg := range mlChan {
			log.Printf("[NET] Received ML message: %+v", msg)
		}
	}()

	// ---- Relay Addresses ----
	relayAddrs, err := relayhelper.GetRelayAddrFromMongo()
	if err != nil {
		log.Printf("Error during get relay addrs: %v", err.Error())
		return
	}
	log.Printf("relayAddrs in Mongo: %+v\n", relayAddrs)

	// ---- Start Peer ----
	p, err := peer.NewPeer(relayAddrs, "user")
	if err != nil {
		log.Printf("Error on NewDepthPeer: %v\n", err.Error())
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := peer.Start(p, ctx); err != nil {
		log.Printf("Error starting peer: %v", err)
		return
	}

	// ---- Kademlia Integration ----
	var kademliaHandler *integration.ComprehensiveKademliaHandler
	if *useKademlia {
		kademliaHandler = integration.NewComprehensiveKademliaHandler()

		log.Printf("PID of node is %v", p.Host.ID().String())
		err = kademliaHandler.InitializeNode(
			"nodeid.db",
			p.Host.ID().String(),
			"./kademlia_relay.db",
		)
		if err != nil {
			log.Printf("Failed to initialize Kademlia: %v", err)
		} else {
			log.Println("✓ Kademlia integration initialized")

			// ---- Bootstrap ----
			bootstrapNodes, err := helpers.ParseBootstrapFlags(*nodeidHex, *pid)
			if err != nil {
				log.Printf("Error: %v", err)
			} else if len(bootstrapNodes) > 0 {
				log.Println("🚀 Starting Kademlia bootstrap process...")
				successCount := 0
				for _, addr := range bootstrapNodes {
					nodeBytes, err := hex.DecodeString(addr.NodeID)
					if err != nil {
						log.Printf("❌ Invalid NodeID for %s: %v", addr, err)
						continue
					}
					if err := addPeerToRoutingTable(kademliaHandler, addr.PeerID, nodeBytes); err != nil {
						log.Printf("❌ Bootstrap failed for %s: %+v", addr, err)
					} else {
						successCount++
						log.Printf("✅ Bootstrap successful for %s", addr)
					}
				}

				if successCount > 0 {
					log.Printf("✓ Bootstrap completed with %d/%d successful connections",
						successCount, len(bootstrapNodes))

					// Ping bootstrap peers
					for _, addr := range bootstrapNodes {
						if err := pingPeer(p, ctx, addr.PeerID); err != nil {
							log.Printf("pingPeer err: %v", err.Error())
						}
					}
				} else {
					log.Println("⚠️  Bootstrap failed for all nodes - running in isolated mode")
				}
			}

			// ---- Add target peer ----
			if *sendReq && *pid != "" && len(nodeidBytes) > 0 {
				if err := addPeerToRoutingTable(kademliaHandler, *pid, nodeidBytes); err != nil {
					log.Printf("Failed to add target peer: %v", err)
				} else {
					log.Printf("✓ Added target peer to routing table: %s", *pid)
				}
			}

			// ---- Store test embeddings ----
			storeTestEmbeddings(kademliaHandler, nodeidBytes)

			// ---- Print routing stats ----
			stats := kademliaHandler.GetNodeStatistics()
			routingInfo := kademliaHandler.GetRoutingInfo()
			log.Printf("📊 Node Statistics: %+v", stats)
			log.Printf("🗺️  Routing table contains %d peers", len(routingInfo))
		}
	}

	// ---- Conditionally send request ----
	if *sendReq {
		if *pid == "" {
			log.Fatal("You must provide a -pid value when using -sendreq")
		}
		if len(nodeidBytes) == 0 {
			log.Fatal("You must provide a -nodeid value when using -sendreq")
		}

		params := models.PingRequest{
			Type:           "GET",
			Route:          "ping",
			SenderPeerID:   p.Host.ID().String(),
			ReceiverPeerID: *pid,
			Timestamp:      time.Now().Unix(),
		}

		var requestBody interface{}
		if *useKademlia && kademliaHandler != nil {
			requestBody, params.Route = helpers.BuildKademliaRequest(kademliaHandler, nodeidBytes)
		} else {
			requestBody = map[string]interface{}{
				"message":       "Hello from relay peer",
				"timestamp":     time.Now().Unix(),
				"kademlia_used": false,
			}
		}

		resp, err := helpers.SendJSON(p, ctx, *pid, params, requestBody)
		if err != nil {
			log.Printf("Error sending request: %v", err)
		} else if len(resp) > 0 {
			var respDec any
			json.Unmarshal(resp, &respDec)
			log.Printf("Response: %+v", respDec)
		} else {
			log.Println("Received empty response")
		}
	}

	<-ctx.Done()
}

//
// ---- Helpers ----
//


func addPeerToRoutingTable(handler *integration.ComprehensiveKademliaHandler, pid string, nodeID []byte) error {
	log.Printf("Adding peer to routing table: PID=%s NID len=%v", pid, len(nodeID))
	return handler.AddPeerToRoutingTable(types.PeerInfo{NodeID: nodeID, PeerID: pid})
}

func storeTestEmbeddings(handler *integration.ComprehensiveKademliaHandler, nodeidBytes []byte) {
	testFiles := []genmodels.ClusterFile{
		{
			Filename: "document1.pdf",
			Metadata: genmodels.FileMetadata{
				Name:         "document1.pdf",
				CreatedAt:    time.Now().Format(time.RFC3339),
				LastModified: time.Now().Format(time.RFC3339),
				FileSize:     1024.5,
				UpdatedAt:    time.Now().Format(time.RFC3339),
			},
			Embedding: []float64{0.1, 0.2, 0.3, 0.4, 0.5},
		},
		{
			Filename: "image1.jpg",
			Metadata: genmodels.FileMetadata{
				Name:         "image1.jpg",
				CreatedAt:    time.Now().Format(time.RFC3339),
				LastModified: time.Now().Format(time.RFC3339),
				FileSize:     2048.7,
				UpdatedAt:    time.Now().Format(time.RFC3339),
			},
			Embedding: []float64{0.9, 0.1, 0.0, 0.0, 0.0},
		},
	}

	for _, file := range testFiles {
		if err := handler.StoreEmbedding(nodeidBytes, file.Embedding); err != nil {
			log.Printf("Failed to store embedding for %s: %v", file.Filename, err)
		} else {
			log.Printf("✓ Stored embedding for file: %s", file.Filename)
		}
	}
}

func pingPeer(p *models.UserPeer, ctx context.Context, pid string) error {
	params := models.PingRequest{
		Type:           "GET",
		Route:          "ping",
		ReceiverPeerID: pid,
		Timestamp:      time.Now().Unix(),
	}
	resp, err := helpers.SendJSON(p, ctx, pid, params, map[string]interface{}{})
	if err != nil {
		log.Printf("Ping failed for peer %s: %v", pid, err.Error())
		return err
	}
	log.Printf("Ping response from peer %s: %s", pid, string(resp))
	return nil
}