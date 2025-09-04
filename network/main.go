package main

import (
	"context"
	"database/sql"
	"encoding/hex"
	"encoding/json"

	"final/backend/pkg/identity"
	genmodels "final/network/RelayFinal/pkg/generalpeer/models"
	"final/network/RelayFinal/pkg/generalpeer/ws"
	"final/network/RelayFinal/pkg/network/helpers"
	relayhelper "final/network/RelayFinal/pkg/relay/helpers"
	"final/network/RelayFinal/pkg/relay/models"
	"final/network/RelayFinal/pkg/relay/peer"
	"flag"
	"log"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"final/backend/pkg/integration"
	"final/backend/pkg/types"
)

func main() {
	// ---- Flags ----
	findval := flag.Bool("findval", false, "Search for query embedding")
	pid := flag.String("pid", "", "Target peer ID to send request to (comma sep)")
	nodeidHex := flag.String("nodeid", "", "Node ID for target peer (hex, comma sep)")
	store := flag.Bool("store", false, "Upload a file to the network")
	flag.Parse()

	// ---- Decode nodeid ----
	// var nodeidBytes []byte // this is for the target node id
	// if *nodeidHex != "" {
	// 	var err error
	// 	nodeidBytes, err = hex.DecodeString(*nodeidHex)
	// 	if err != nil {
	// 		log.Fatalf("invalid bootstrap NodeID: %v", err)
	// 	}
	// }
	SourceNodeID, err := identity.LoadOrCreateNodeID("")

	if err != nil {
		log.Println("Could not load/create the node id")
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
	// var kademliaHandler *integration.ComprehensiveKademliaHandler

	kademliaHandler := integration.NewComprehensiveKademliaHandler()

	// node initialization and bootstrap
	selfNodeID, err := kademliaHandler.InitializeNode(
		p.Host.ID().String(),
		"./nodeid_embedding_map.db",
	)
	if err != nil {
		log.Printf("Failed to initialize Kademlia: %v", err)
		return
	}

	decSelfNodeID := hex.EncodeToString(selfNodeID)
	err  = relayhelper.UpsertNode(decSelfNodeID, p.Host.ID().String())
	if(err != nil){
		log.Printf("Error in upserting node to mongo: %v \n", err.Error())
	} else {
		log.Println("✓ Kademlia integration initialized")

		// ---- Bootstrap ---- only to be done when nodeid and pid provided
		if *nodeidHex == "" || *pid == "" {
			log.Println("No bootstrap nodes provided, skipping bootstrap")
		} else {
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
						if err := pingPeer(p, ctx, addr.NodeID); err != nil {
							log.Printf("pingPeer err: %v", err.Error())
						}
					}
				} else {
					log.Println("⚠️  Bootstrap failed for all nodes - running in isolated mode")
				}
			}
		}

		// ---- Add target peer ----
		// if *sendReq && *pid != "" && len(nodeidBytes) > 0 {
		// 	if err := addPeerToRoutingTable(kademliaHandler, *pid, nodeidBytes); err != nil {
		// 		log.Printf("Failed to add target peer: %v", err)
		// 	} else {
		// 		log.Printf("✓ Added target peer to routing table: %s", *pid)
		// 	}
		// }

		// ---- Store test embeddings ----
		// nid, err := identity.LoadOrCreateNodeID("")
		// if err!=nil{
		// 	log.Printf("Node Id not created/loaded")
		// }else{
		// 	storeTestEmbeddings(kademliaHandler, nid)
		// }

		// ---- Print routing stats ----
		stats := kademliaHandler.GetNodeStatistics()
		routingInfo := kademliaHandler.GetRoutingInfo()
		log.Printf("📊 Node Statistics: %+v", stats)
		log.Printf("🗺️  Routing table contains %d peers", len(routingInfo))
	}

	// ---- Conditionally send request ----
	if *findval {
		// pid is not required while sending a request
		// if *pid == "" {
		// 	log.Fatal("You must provide a -pid value when using -sendreq")
		// }
		test_embedding := []float64{0.1, 0.2, 0.3, 0.4, 0.5}

		// get the node id of the closest peer in the nodeid_embedding_map.db
		threshold := 0;
		limit := 1;
		var TargetNodeID []byte
		targetnodeids, err := kademliaHandler.Node.FindSimilar(test_embedding, threshold, limit)
		if err != nil{
			log.Println("Could not find the target node id")
		}
		// then find the peer id of the node closest to the target node from the routing table
		TargetNodeID = targetnodeids[0].Key
		nextNode := kademliaHandler.Node.RoutingTable.FindClosest(TargetNodeID, limit)

		
		// params := models.PingRequest{
		// 	Type:           "GET",
		// 	Route:          "ping",
		// 	SenderPeerID:   p.Host.ID().String(),
		// 	ReceiverPeerID: *pid, // t
		// 	Timestamp:      time.Now().Unix(),
		// }

		params := models.EmbeddingSearchRequest{
			Type:           "GET",
			Route:          "find_value",
			SourceNodeID:  SourceNodeID,
			SourcePeerID: p.Host.ID().String(),
			TargetNodeID: TargetNodeID,
			ReceiverPeerID: nextNode.PeerID,
		}

		var requestBody interface{}
		if kademliaHandler != nil {
			requestBody, params.Route = helpers.BuildKademliaFindValueRequest(kademliaHandler, SourceNodeID, test_embedding)
		} else {
			requestBody = map[string]interface{}{
				"message":       "Hello from relay peer",
				"timestamp":     time.Now().Unix(),
				"kademlia_used": false,
			}
		}

		resp, err := helpers.SendJSON(p, ctx, nextNode.PeerID, params, requestBody)
		if err != nil {
			log.Printf("Error sending request: %v", err)
		} else if len(resp) > 0 {
			var respDec any
			json.Unmarshal(resp, &respDec)
			log.Printf("Response: %+v", respDec)
		} else {
			log.Println("Received empty response")
		}
		// while depth!=4 && found!=true get resp
	}

	if *store {
		// if kademliaHandler == nil {
		// 	log.Println("Node not initialized with Kademlia, cannot store embeddings")
		// }
		// test_embedding := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
		// params := models.PingRequest{
		// 	Type:           "GET",
		// 	Route:          "ping",
		// 	SenderPeerID:   p.Host.ID().String(),
		// 	ReceiverPeerID: *pid,
		// 	Timestamp:      time.Now().Unix(),
		// }
	}

	<-ctx.Done()

	// Save routing table to DB before exit
	if kademliaHandler != nil {
		if err := saveRoutingTableToDB(kademliaHandler); err != nil {
			log.Printf("Failed to save routing table: %v", err)
		} else {
			log.Println("✓ Routing table saved to DB.")
		}
	}
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

func saveRoutingTableToDB(handler *integration.ComprehensiveKademliaHandler) error {
	routingDBPath := "routing_table.db"
	db, err := sql.Open("sqlite3", routingDBPath)
	if err != nil {
		return err
	}
	defer db.Close()

	stmt, err := db.Prepare(`INSERT OR REPLACE INTO routing_table (node_id, peer_id) VALUES (?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	routingInfo := handler.GetRoutingInfo()
	for _, peer := range routingInfo {
		nodeIDHex := hex.EncodeToString(peer.NodeID)
		_, err := stmt.Exec(nodeIDHex, peer.PeerID)
		if err != nil {
			return err
		}
	}
	return nil
}

func getClosestNodeIDFromDB(dbPath string, targetEmbedding []float64) (string, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return "", err
	}
	defer db.Close()

	rows, err := db.Query("SELECT node_id, embedding FROM nodeid_embedding_map")
	if err != nil {
		return "", err
	}
	defer rows.Close()

	var closestNodeID string
	minDist := 1e12 // large initial value

	for rows.Next() {
		var nodeID string
		var embeddingJSON string
		if err := rows.Scan(&nodeID, &embeddingJSON); err != nil {
			return "", err
		}

		var embedding []float64
		if err := json.Unmarshal([]byte(embeddingJSON), &embedding); err != nil {
			continue // skip invalid embeddings
		}

		if len(embedding) != len(targetEmbedding) {
			continue // skip if dimensions mismatch
		}

		dist := 0.0
		for i := range embedding {
			diff := embedding[i] - targetEmbedding[i]
			dist += diff * diff
		}

		if dist < minDist {
			minDist = dist
			closestNodeID = nodeID
		}
	}

	if closestNodeID == "" {
		return "", sql.ErrNoRows
	}
	return closestNodeID, nil
}

