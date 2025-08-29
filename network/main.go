package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	"time"

	// Your actual network imports based on the GitHub structure
	genmodels "final/network/RelayFinal/pkg/generalpeer/models"
	"final/network/RelayFinal/pkg/generalpeer/ws"
	"final/network/RelayFinal/pkg/relay/helpers"
	"final/network/RelayFinal/pkg/relay/models"
	"final/network/RelayFinal/pkg/relay/peer"

	// Import your Kademlia integration wrapper
	"final/backend/pkg/integration"
	
	// Kademlia helpers for node ID conversion
	kadhelpers "final/backend/pkg/helpers"
)

func main() {
	var mlChan chan genmodels.ClusterWrapper

	// Define flags
	sendReq := flag.Bool("sendreq", false, "Whether to send request to target peer")
	pid := flag.String("pid", "", "Target peer ID to send request to (used only if sendreq=true)")
	useKademlia := flag.Bool("kademlia", false, "Use Kademlia for embedding search")
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
	}
	log.Printf("relayAddrs in Mongo: %+v\n", relayAddrs)

	// Start Depth Peer using your actual NewDepthPeer function
	p, err := peer.NewDepthPeer(relayAddrs)
	if err != nil {
		log.Printf("Error on NewDepthPeer: %v\n", err.Error())
	}

	ctx := context.Background()
	peer.Start(p, ctx)

	// ===== KADEMLIA INTEGRATION =====
	var kademliaHandler *integration.ComprehensiveKademliaHandler
	
	if *useKademlia {
		// Initialize Kademlia handler
		kademliaHandler = integration.NewComprehensiveKademliaHandler()
		
		// Use the actual Host ID from your DepthPeer
		err = kademliaHandler.InitializeNode(
			"relay-node-"+time.Now().String(),
			p.Host.ID().String(), // Use the actual libp2p Host ID
			"./kademlia_relay.db",
		)
		if err != nil {
			log.Printf("Failed to initialize Kademlia: %v", err)
		} else {
			log.Println("✓ Kademlia integration initialized")
			
			// Store some test embeddings using your actual models
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
		}
	}

	// Conditionally send request if flag is set
	if *sendReq {
		if *pid == "" {
			log.Fatal("You must provide a -pid value when using -sendreq")
		}

		// Create request using your actual PingRequest model
		params := models.PingRequest{
			Type:           "GET",
			Route:          "ping",
			ReceiverPeerID: *pid,
			Timestamp:      time.Now().Unix(),
		}
		
		// Create request body based on whether Kademlia is used
		var requestBody interface{}
		
		if *useKademlia && kademliaHandler != nil {
			log.Println("🔍 Creating Kademlia embedding search request...")
			
			// Create input Message using your actual genmodels.Message structure
			inputMessage := genmodels.Message{
				Type:          "embedding_search",
				QueryEmbed:    []float64{0.1, 0.2, 0.3, 0.4, 0.5},
				Depth:         0,
				CurrentPeerID: 0, // Current peer (will be set by system)
				NextPeerID:    0, // Target peer (to be determined by Kademlia)
				FileMetadata: genmodels.FileMetadata{
					Name:         "query_document.pdf",
					CreatedAt:    time.Now().Format(time.RFC3339),
					LastModified: time.Now().Format(time.RFC3339),
					FileSize:     512.0,
					UpdatedAt:    time.Now().Format(time.RFC3339),
				},
				IsProcessed:   false,
				Found:         false,
			}

			// Process through Kademlia wrapper
			targetNodeID := kadhelpers.HashNodeIDFromString(*pid)
			kademliaResponse, err := kademliaHandler.ProcessEmbeddingRequestWrapper(
				inputMessage.QueryEmbed,
				targetNodeID,
				inputMessage.Type,
				0.8, // Threshold
				10,  // Results count
			)

			// Create output message with Kademlia results
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

			// Set request body with both input and output using your models
			requestBody = map[string]interface{}{
				"input_message":  inputMessage,
				"output_message": outputMessage,
				"kademlia_used":  true,
				"timestamp":      time.Now().Unix(),
			}

			// Update route to indicate Kademlia usage
			params.Route = "kademlia_search"

		} else {
			// Regular request body (original behavior)
			requestBody = map[string]interface{}{
				"message":       "Hello from relay peer",
				"timestamp":     time.Now().Unix(),
				"kademlia_used": false,
			}
		}

		// Marshal both params and body
		jsonParams, _ := json.Marshal(params)
		bodyJson, _ := json.Marshal(requestBody)

		// Send request using your actual peer.Send function
		resp, err := peer.Send(p, ctx, *pid, jsonParams, bodyJson)
		if err != nil {
			log.Printf("Error sending request: %v", err)
		} else {
			var respDec any
			json.Unmarshal(resp, &respDec)
			log.Printf("Response: %+v", respDec)
		}
	}

	// Block main goroutine
	select {}
}
