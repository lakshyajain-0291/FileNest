package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	"strings"
	"time"

	// Your actual network imports based on the GitHub structure
	genmodels "final/network/RelayFinal/pkg/generalpeer/models"
	"final/network/RelayFinal/pkg/generalpeer/ws"
	"final/network/RelayFinal/pkg/relay/helpers"
	"final/network/RelayFinal/pkg/relay/models"
	"final/network/RelayFinal/pkg/relay/peer"

	// Import your Kademlia integration wrapper
	"final/backend/pkg/integration"
	"final/backend/pkg/types"

	// Kademlia helpers for node ID conversion
	kadhelpers "final/backend/pkg/helpers"
)

type IDs struct{
	PeerID string
	NodeID string
}

func main() {
	var mlChan chan genmodels.ClusterWrapper

	// Define flags
	sendReq := flag.Bool("sendreq", false, "Whether to send request to target peer")
	pid := flag.String("pid", "", "Target peer ID to send request to (used only if sendreq=true), should be comma sep")
	nodeid := flag.String("nodeid", "", "Node ID for this peer, should be comma sep")
	useKademlia := flag.Bool("kademlia", true, "Use Kademlia for embedding search")
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

	// Get relay addresses from MongoDB
	relayAddrs, err := helpers.GetRelayAddrFromMongo()
	if err != nil {
		log.Printf("Error during get relay addrs: %v", err.Error())
		return
	}
	log.Printf("relayAddrs in Mongo: %+v\n", relayAddrs)

	// Start Depth Peer using your actual NewDepthPeer function
	p, err := peer.NewPeer(relayAddrs, "USER")
	if err != nil {
		log.Printf("Error on NewDepthPeer: %v\n", err.Error())
		return
	}

	ctx := context.Background()
	err = peer.Start(p, ctx)
	if err != nil {
		log.Printf("Error starting peer: %v", err)
		return
	}

	// ===== KADEMLIA INTEGRATION =====
	var kademliaHandler *integration.ComprehensiveKademliaHandler

	if *useKademlia {
		// Initialize Kademlia handler
		kademliaHandler = integration.NewComprehensiveKademliaHandler()

		// Use the actual Host ID from your DepthPeer
		err = kademliaHandler.InitializeNode(
			"nodeid.db",
			p.Host.ID().String(),
			"./kademlia_relay.db",
		)

		if err != nil {
			log.Printf("Failed to initialize Kademlia: %v", err)
		} else {
			log.Println("✓ Kademlia integration initialized")

			// BOOTSTRAP LOGIC - Multiple strategies
			var bootstrapNodes []IDs

			// Strategy 1: Use explicit bootstrap addresses if provided
			nodeIDs := strings.Split(*nodeid, " ")
			peerIDs := strings.Split(*pid, " ")
			if len(nodeIDs) != len(peerIDs) {
				log.Printf("Error: Number of node IDs and peer IDs must match.")
			} else {
				for i := range nodeIDs {
					bootstrapNodes = append(bootstrapNodes, IDs{
						PeerID: peerIDs[i],
						NodeID: nodeIDs[i],
					})
				}
				log.Printf("🔄 Using explicit bootstrap nodes: %v", bootstrapNodes)
			}
			// } else {
			// 	// Strategy 2: Use relay addresses as bootstrap nodes
			// 	bootstrapNodes = relayAddrs
			// 	log.Printf("🔄 Using relay addresses as bootstrap nodes: %v", bootstrapNodes)
			// }

			// Perform bootstrap
			if len(bootstrapNodes) > 0 {
				log.Println("🚀 Starting Kademlia bootstrap process...")
				successCount := 0

				for _, addr := range bootstrapNodes {
					err := bootstrapFromAddress(kademliaHandler, addr.PeerID, []byte(addr.NodeID))
					if err != nil {
						log.Printf("❌ Bootstrap failed for %s: %v", addr, err)
					} else {
						successCount++
						log.Printf("✅ Bootstrap successful for %s", addr)
					}
				}

				if successCount > 0 {
					log.Printf("✓ Bootstrap completed with %d/%d successful connections",
						successCount, len(bootstrapNodes))

					// Perform iterative lookup to discover more peers
					// log.Println("🔍 Performing peer discovery...")
					// _, err := kademliaHandler.IterativeFindNode(
					// 	kadhelpers.HashNodeIDFromString(p.Host.ID().String()))
					// if err != nil {
					// 	log.Printf("Warning: Peer discovery failed: %v", err)
					// } else {
					// 	log.Println("✓ Peer discovery completed")
					// }
				} else {
					log.Println("⚠️  Bootstrap failed for all nodes - running in isolated mode")
				}
			}

			// Add target peer to routing table if provided
			if *sendReq && *pid != "" {
				err := addTargetPeerToRoutingTable(kademliaHandler, *pid)
				if err != nil {
					log.Printf("Failed to add target peer: %v", err)
				} else {
					log.Printf("✓ Added target peer to routing table: %s", (*pid)[:12]+"...")
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
				nodeID := kadhelpers.HashNodeIDFromString(file.Filename)
				err := kademliaHandler.StoreEmbedding(nodeID, file.Embedding)
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

		// Create request using your actual models
		params := models.PingRequest{
			Type:           "GET",
			Route:          "ping",
			ReceiverPeerID: *pid,
			Timestamp:      time.Now().Unix(),
		}

		var requestBody interface{}

		if *useKademlia && kademliaHandler != nil {
			log.Println("🔍 Creating Kademlia embedding search request...")

			// Create input Message
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

			// Process through Kademlia wrapper
			targetNodeID := kadhelpers.HashNodeIDFromString(*pid)
			kademliaResponse, err := kademliaHandler.ProcessEmbeddingRequestWrapper(
				inputMessage.QueryEmbed,
				targetNodeID,
				inputMessage.Type,
				0.8,
				10,
			)

			// Create output message
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

		// Send request
		jsonParams, _ := json.Marshal(params)
		bodyJson, _ := json.Marshal(requestBody)
		log.Println("bodyJson created successfully. ")
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

	// Block main goroutine
	select {}
}

// Helper function to bootstrap from a single address
func bootstrapFromAddress(handler *integration.ComprehensiveKademliaHandler, addr string, NodeID []byte) error {
	// Extract peer ID from multiaddr

	peerIDStr := addr
	nodeID := NodeID

	// Add to routing table
	peerInfo := types.PeerInfo{
		NodeID: nodeID,
		PeerID: peerIDStr,
	}

	return handler.AddPeerToRoutingTable(peerInfo)
}

// Helper function to add target peer to routing table
func addTargetPeerToRoutingTable(handler *integration.ComprehensiveKademliaHandler, pid string) error {
	targetNodeID := kadhelpers.HashNodeIDFromString(pid)
	targetPeer := types.PeerInfo{
		NodeID: targetNodeID,
		PeerID: pid,
	}

	return handler.AddPeerToRoutingTable(targetPeer)
}
