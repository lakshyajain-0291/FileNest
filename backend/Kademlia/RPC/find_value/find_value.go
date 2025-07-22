package findvalue

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
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
	TargetPeerID int       `json:"target_peer_id"`
}

type EmbeddingSearchResponse struct {
	Type          string    `json:"type"`
	QueryEmbed    []float64 `json:"query_embed"`
	Depth         int       `json:"depth"`
	CurrentPeerID int       `json:"current_peer_id"`
	NextPeerID    int       `json:"next_peer_id"`
	FileMetadata  Metadata  `json:"file_metadata"`
	IsProcessed   bool      `json:"is_processed"` //this is to check if the query has the reached the D4 node or not
	Found         bool      `json:"found"` //this is to confirm if the target peer id for a peer at any depth has been found or not
}

type Metadata struct {
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

func FindValue(peerId int) (bool, []FileInfo) {
	var mindistance int
	var metadata []FileInfo

	// TODO: Check the routing table for closest matches
	// and append to metadata

	status := mindistance == 0
	return status, metadata
}

func HandleJSONMessages(s network.Stream) {
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
		var current_peer_id int // need to pass the peer id of the depth peer

		if msg.QueryType == "search" {
			// Send acknowledgment
			status, fileinfo := FindValue(msg.TargetPeerID) // use the find_node rpc to fetch the metadata of the peer to which the request has to be forwarded
		
				
			for _, file := range fileinfo {
				if current_depth!=4{
					ack := EmbeddingSearchResponse{Type: msg.QueryType, QueryEmbed: msg.Embed, Depth: current_depth,
					CurrentPeerID: current_peer_id, NextPeerID: file.PeerID(), IsProcessed: false, Found: status}
					encoder.Encode(ack)
				}else {
					// will iterate the following block of code over the number of records in the database
					metadata := Metadata{} //need to fetch the indexed file metadata from the db.
					ack := EmbeddingSearchResponse{Type: msg.QueryType, QueryEmbed: msg.Embed, Depth: current_depth,
						CurrentPeerID: current_peer_id, FileMetadata: metadata, IsProcessed: true, Found: status}
					encoder.Encode(ack)
				}
			} 
			
		} else {
			// Send acknowledgment
			status, fileinfo := FindValue(msg.TargetPeerID)
			
			for _, file := range fileinfo{
				if current_depth != 4 {
					ack := EmbeddingSearchResponse{Type: msg.QueryType, QueryEmbed: msg.Embed, Depth: current_depth,
					CurrentPeerID: current_peer_id, NextPeerID: file.PeerID(), IsProcessed: false, Found: status}
					encoder.Encode(ack)
				} else {
					// will iterate the following block of code over the number of records in the database
					// need to index the file under the peer id. will use the store rpc .
					ack := EmbeddingSearchResponse{} // need to modify the api so that acknowledgment for updation of file can be included
					encoder.Encode(ack)
				}				
			}

		}
	}
}
