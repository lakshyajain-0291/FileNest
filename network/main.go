package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"flag"

	// "fmt"
	"log"
	"strings"
	"time"

	genmodels "final/network/RelayFinal/pkg/generalpeer/models"
	"final/network/RelayFinal/pkg/generalpeer/ws"
	"final/network/RelayFinal/pkg/relay/helpers"
	"final/network/RelayFinal/pkg/relay/models"
	"final/network/RelayFinal/pkg/relay/peer"

	"final/backend/pkg/identity"
	"final/backend/pkg/integration"
	"final/backend/pkg/types"
)

type IDs struct {
	PeerID string
	NodeID string // hex string for CLI/logging
}

func main() {
	var mlChan chan genmodels.ClusterWrapper

	// Define flags
	sendReq := flag.Bool("sendreq", false, "Whether to send request to target peer")
	pid := flag.String("pid", "", "Target peer ID to send request to (comma sep)")
	nodeidHex := flag.String("nodeid", "", "Node ID for this peer (hex, comma sep)")
	useKademlia := flag.Bool("kademlia", true, "Use Kademlia for embedding search")
	flag.Parse()

	// Decode nodeid flag into raw bytes once
	var nodeidBytes []byte
	if *nodeidHex != "" {
		var err error
		nodeidBytes, err = hex.DecodeString(*nodeidHex)
		if err != nil {
			log.Fatalf("invalid bootstrap NodeID: %v", err)
		}
	}

	mlTransport := ws.NewWebSocketTransport(":8081")
	defer mlTransport.Close()

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

	// Get relay addresses from MongoDB
	relayAddrs, err := helpers.GetRelayAddrFromMongo()
	if err != nil {
		log.Printf("Error during get relay addrs: %v", err.Error())
		return
	}
	log.Printf("relayAddrs in Mongo: %+v\n", relayAddrs)

	// Start Depth Peer
	p, err := peer.NewPeer(relayAddrs, "user")
	if err != nil {
		log.Printf("Error on NewDepthPeer: %v\n", err.Error())
		return
	}

	ctx := context.Background()
	if err := peer.Start(p, ctx); err != nil {
		log.Printf("Error starting peer: %v", err)
		return
	}

	// ===== KADEMLIA INTEGRATION =====
	var kademliaHandler *integration.ComprehensiveKademliaHandler

	if *useKademlia {
		kademliaHandler = integration.NewComprehensiveKademliaHandler()

		log.Printf("PID of node is %v", p.Host.ID().String())
		err = kademliaHandler.InitializeNode(
			p.Host.ID().String(),
			"./kademlia_relay.db",
		)
		if err != nil {
			log.Printf("Failed to initialize Kademlia: %v", err)
		} else {
			log.Println("✓ Kademlia integration initialized")

			// BOOTSTRAP LOGIC
			var bootstrapNodes []IDs

			var nodeIDs []string
			if *nodeidHex != "" {
				nodeIDs = strings.Split(*nodeidHex, ",")
			}

			var peerIDs []string
			if *pid != "" {
				peerIDs = strings.Split(*pid, ",")
			}

			if len(nodeIDs) != len(peerIDs) {
				log.Printf("Error: Number of node IDs and peer IDs must match.")
			} else {
				for i := range nodeIDs {
					bootstrapNodes = append(bootstrapNodes, IDs{
						PeerID: peerIDs[i],
						NodeID: nodeIDs[i], // hex string
					})
				}
				log.Printf("🔄 Using explicit bootstrap nodes: %+v", bootstrapNodes)
			}

			// Perform bootstrap
			if len(bootstrapNodes) > 0 {
				log.Println("🚀 Starting Kademlia bootstrap process...")
				successCount := 0

				for _, addr := range bootstrapNodes {
					nodeBytes, err := hex.DecodeString(addr.NodeID)
					if err != nil {
						log.Printf("❌ Invalid NodeID for %s: %v", addr, err)
						continue
					}
					err = bootstrapFromAddress(kademliaHandler, addr.PeerID, nodeBytes)
					if err != nil {
						log.Printf("❌ Bootstrap failed for %s: %+v", addr, err)
					} else {
						successCount++
						log.Printf("✅ Bootstrap successful for %s", addr)
					}
				}

				if successCount > 0 {
					log.Printf("✓ Bootstrap completed with %d/%d successful connections",
						successCount, len(bootstrapNodes))

					// ping the bootstrapped nodes.
					for _, peerid := range peerIDs {
						pingPeer(p, ctx, peerid)
					}
				} else {
					log.Println("⚠️  Bootstrap failed for all nodes - running in isolated mode")
				}
			}

			// Add target peer to routing table if provided
			if *sendReq && *pid != "" {
				if len(nodeidBytes) == 0 {
					log.Fatal("No NodeID provided for target peer")
				}
				err := addTargetPeerToRoutingTable(kademliaHandler, *pid, nodeidBytes)
				if err != nil {
					log.Printf("Failed to add target peer: %v", err)
				} else {
					log.Printf("✓ Added target peer to routing table: %s", *pid)
				}
			}

			// Store test embeddings
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
				// just use filename hashed into NodeID
				nodeid, err := identity.LoadOrCreateNodeID("")
				if err != nil {
					log.Printf("Failed to load/create NodeID: %v", err)
					continue
				}	
				err = kademliaHandler.StoreEmbedding(nodeid, file.Embedding)
				if err != nil {
					log.Printf("Failed to store embedding for %s: %v", file.Filename, err)
				} else {
					log.Printf("✓ Stored embedding for file: %s", file.Filename)
				}
			}

			// Display routing table statistics
			stats := kademliaHandler.GetNodeStatistics()
			routingInfo := kademliaHandler.GetRoutingInfo()
			log.Printf("📊 Node Statistics: %+v", stats)
			log.Printf("🗺️  Routing table contains %d peers", len(routingInfo))
		}
	}

	// Conditionally send request if flag is set
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
			ReceiverPeerID: *pid,
			Timestamp:      time.Now().Unix(),
		}

		var requestBody interface{}

		if *useKademlia && kademliaHandler != nil {
			log.Println("🔍 Creating Kademlia embedding search request...")

			inputMessage := genmodels.Message{
				Type:          "embedding_search",
				QueryEmbed:    []float64{0.1, 0.2, 0.3, 0.4, 0.5},
				Depth:         0,
				CurrentPeerID: 0,
				NextPeerID:    0,
				FileMetadata: genmodels.FileMetadata{
					Name:         "query_document.pdf",
					CreatedAt:    time.Now().Format(time.RFC3339),
					LastModified: time.Now().Format(time.RFC3339),
					FileSize:     512.0,
					UpdatedAt:    time.Now().Format(time.RFC3339),
				},
				IsProcessed: false,
				Found:       false,
			}

			kademliaResponse, err := kademliaHandler.ProcessEmbeddingRequestWrapper(
				inputMessage.QueryEmbed,
				nodeidBytes,
				inputMessage.Type,
				0.8,
				10,
			)

			outputMessage := genmodels.Message{
				Type:          inputMessage.Type,
				QueryEmbed:    inputMessage.QueryEmbed,
				Depth:         inputMessage.Depth + 1,
				CurrentPeerID: inputMessage.CurrentPeerID,
				NextPeerID:    inputMessage.NextPeerID,
				FileMetadata:  inputMessage.FileMetadata,
				IsProcessed:   err == nil,
				Found:         false,
			}

			if err != nil {
				log.Printf("❌ Kademlia processing failed: %v", err)
			} else {
				outputMessage.Found = kademliaResponse.Found
				log.Printf("✅ Kademlia processed successfully:")
				log.Printf("   Found: %t", kademliaResponse.Found)
				log.Printf("   Query Type: %s", kademliaResponse.QueryType)
				if kademliaResponse.NextNodeID != nil {
					log.Printf("   Next Node: %x", kademliaResponse.NextNodeID[:8])
				}
			}

			requestBody = map[string]interface{}{
				"input_message":  inputMessage,
				"output_message": outputMessage,
				"kademlia_used":  true,
				"timestamp":      time.Now().Unix(),
			}

			params.Route = "kademlia_search"

		} else {
			requestBody = map[string]interface{}{
				"message":       "Hello from relay peer",
				"timestamp":     time.Now().Unix(),
				"kademlia_used": false,
			}
		}

		jsonParams, _ := json.Marshal(params)
		bodyJson, _ := json.Marshal(requestBody)
		resp, err := peer.Send(p, ctx, *pid, jsonParams, bodyJson)
		if err != nil {
			log.Printf("Error sending request: %v", err)
		} else {
			var respDec any
			if len(resp) > 0 {
				json.Unmarshal(resp, &respDec)
				log.Printf("Response: %+v", respDec)
			} else {
				log.Println("Received empty response")
			}
		}
	}

	select {}
}

// Bootstrap helper
func bootstrapFromAddress(handler *integration.ComprehensiveKademliaHandler, pid string, nodeID []byte) error {
	log.Printf("bootstrapFromAddress: NID len: %v", len(nodeID))
	peerInfo := types.PeerInfo{
		NodeID: nodeID,
		PeerID: pid,
	}
	return handler.AddPeerToRoutingTable(peerInfo)
}

// Add target peer helper
func addTargetPeerToRoutingTable(handler *integration.ComprehensiveKademliaHandler, pid string, nodeID []byte) error {
	log.Printf("main.go targetnodeid length: %v", len(nodeID))
	targetPeer := types.PeerInfo{
		NodeID: nodeID,
		PeerID: pid,
	}
	return handler.AddPeerToRoutingTable(targetPeer)
}

// Ping helper
func pingPeer(p *models.UserPeer, ctx context.Context, pid string) error {
	params := models.PingRequest{
		Type:           "GET",
		Route:          "ping",
		ReceiverPeerID: pid,
		Timestamp:      time.Now().Unix(),
	}
	paramsJson, _ := json.Marshal(params)
	bodyJson, _ := json.Marshal(map[string]interface{}{})
	resp, err := peer.Send(p, ctx, pid, paramsJson, bodyJson)
	if err != nil {
		log.Printf("Ping failed for peer %s: %v", pid, err.Error())
		return err
	}
	log.Printf("Ping response from peer %s: %s", pid, string(resp))
	return nil
}
