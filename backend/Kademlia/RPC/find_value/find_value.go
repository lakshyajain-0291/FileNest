package findvalue

import (
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"time"

	store "dht/RPC/store"
	routing_table "dht/routing_table"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
)

type FileInfo interface {
	PeerID() int
	IPv4() string
	Port() int
}

type EmbeddingSearchRequest struct {
	Source       string    `json:"source"`
	SourceID     int       `json:"source_id"`
	Embed        []float64 `json:"embed"`
	PrevDepth    int       `json:"prev_depth"`
	QueryType    string    `json:"query_type"`
	Threshold    float64   `json:"threshold"`
	ResultsCount int       `json:"results_count"`
	TargetPeerID int       `json:"target_peer_id"` //the target peer id is required at all times to find the nearest nodes
}

type EmbeddingSearchResponse struct {
	Type          string    `json:"type"`
	QueryEmbed    []float64 `json:"query_embed"`
	Depth         int       `json:"depth"`
	CurrentPeerID int       `json:"current_peer_id"`
	NextPeerID    int       `json:"next_peer_id"`
	FileMetadata  store.Record  `json:"file_metadata"` // using the record struct instead of metadata as it provides all the required info
	IsProcessed   bool      `json:"is_processed"` //this is to check if the query has the reached the D4 node or not
	Found         bool      `json:"found"`        //this is to confirm if the target peer id for a peer at any depth has been found or not
}

type Metadata struct {
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

func intToPeerID(id int) peer.ID {
	// Convert int to string, then to peer.ID
	return peer.ID(strconv.Itoa(id))
}

func FindValue(rt *routing_table.RoutingTable, targetPeerId int) (bool, []*routing_table.Contact) {
	// TODO: Check the routing table for closest matches
	targetPeerID := intToPeerID(targetPeerId)
	// use findClosestPeers to find the closest peers and add a check to ensure they are above the threshold. or check with abhinav if this is being done on the frontend
	candidates := rt.FindClosestPeers(targetPeerID, routing_table.BucketSize)
	found := len(candidates) > 0

	return found, candidates
}
func HandleJSONMessages(s network.Stream, current_peer_id int, rt *routing_table.RoutingTable) {
	defer s.Close()

	decoder := json.NewDecoder(s)
	encoder := json.NewEncoder(s)

	for {
		var msg EmbeddingSearchRequest
		err := decoder.Decode(&msg)
		if err != nil {
			fmt.Printf("Stream closed: %v\n", err)
			return
		}

		fmt.Printf("Intercepted JSON message: %+v\n", msg)

		current_depth := msg.PrevDepth + 1

		if msg.QueryType == "search" {
			// Send acknowledgment
			status, fileinfo := FindValue(rt, msg.TargetPeerID) 

			for _, file := range fileinfo {
				nextpeerid, err := strconv.Atoi(string((file.ID)))
				if err != nil {
					log.Printf("Error converting peer ID to int: %v", err)
					continue
				}
				if current_depth != 4 {
					ack := EmbeddingSearchResponse{Type: msg.QueryType, QueryEmbed: msg.Embed, Depth: current_depth,
						CurrentPeerID: current_peer_id, NextPeerID: nextpeerid, IsProcessed: false, Found: status}
					encoder.Encode(ack)
				} else {
					// will iterate the following block of code over the number of records in the database
					records, err := store.ReadAll("Filenest")
					if err != nil {
						fmt.Println("There was an error reading the stored values: %w", err)
					}
					for _, record := range records{
						// metadata := Metadata{} //need to fetch the indexed file metadata from the db.
						ack := EmbeddingSearchResponse{Type: msg.QueryType, QueryEmbed: msg.Embed, Depth: current_depth,
							CurrentPeerID: current_peer_id, FileMetadata: record, IsProcessed: true, Found: status}
						encoder.Encode(ack)
					}
				}
			}

		} else {
			// Send acknowledgment
			status, fileinfo := FindValue(rt, msg.TargetPeerID)

			for _, file := range fileinfo {
				nextpeerid, err := strconv.Atoi(string((file.ID)))
				if err != nil {
					log.Printf("Error converting peer ID to int: %v", err)
					continue
				}
				if current_depth != 4 {
					ack := EmbeddingSearchResponse{Type: msg.QueryType, QueryEmbed: msg.Embed, Depth: current_depth,
						CurrentPeerID: current_peer_id, NextPeerID: nextpeerid, IsProcessed: false, Found: status}
					encoder.Encode(ack)
				} else {
					// need to index the file under the peer id. will use the store api .
					store.Store("Filenest", current_peer_id, msg.Embed, file.Address.String())
					ack := EmbeddingSearchResponse{Type: msg.QueryType, QueryEmbed: msg.Embed, Depth: current_depth,
						CurrentPeerID: current_peer_id, IsProcessed: true, Found: status}
					encoder.Encode(ack)
				}
			}

		}
	}
}
