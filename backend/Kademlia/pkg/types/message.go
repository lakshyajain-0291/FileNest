package types

// contains all the structs for the messages used in the Kademlia protocol

type EmbeddingSearchRequest struct {
	SourceNodeID []byte    `json:"source_id"`
	SourcePeerID string    `json:"source_peer_id"`
	QueryEmbed   []float64 `json:"embed"`
	Depth        int       `json:"prev_depth"`
	QueryType    string    `json:"query_type"`
	Threshold    float64   `json:"threshold"`
	ResultsCount int       `json:"results_count"`
	TargetNodeID []byte    `json:"target_node_id"`
}

type EmbeddingSearchResponse struct {
	QueryType    string    `json:"type"`
	QueryEmbed   []float64 `json:"query_embed"`
	Depth        int       `json:"depth"`
	SourceNodeID []byte    `json:"source_node_id"`
	SourcePeerID string    `json:"source_peer_id"`
	NextNodeID   []byte    `json:"next_node_id"`
	Found        bool      `json:"found"`
	FileEmbed    []float64 `json:"file_embed"`
}

//to be modified with info about receiver
type PingRequest struct {
	SenderNodeID []byte `json:"sender_id"`
	SenderPeerID string `json:"sender_addr"`
	Timestamp    int64  `json:"timestamp"`
}

type PingResponse struct {
	SenderNodeID []byte `json:"sender_id"`
	SenderPeerID string `json:"sender_addr"`
	Timestamp    int64  `json:"timestamp"`
	Success      bool   `json:"success"`
}

type PeerInfo struct {
	NodeID []byte `json:"node_id"`
	PeerID string `json:"peer_id"`
}

// Add these to your existing messages.go file

//sent to kademlia node
type FindNodeRequest struct {
    SenderNodeID []byte `json:"sender_node_id"`
    SenderPeerID string `json:"sender_peer_id"`
    TargetID     []byte `json:"target_id"`    // The NodeID we want to reach
    Timestamp    int64  `json:"timestamp"`
}

//sent back to user for relaying
type FindNodeResponse struct {
    SenderNodeID []byte     `json:"sender_node_id"`
    SenderPeerID string     `json:"sender_peer_id"`
    ClosestNodes []PeerInfo `json:"closest_nodes"`  // K closest nodes to TargetID
    Timestamp    int64      `json:"timestamp"`
    Success      bool       `json:"success"`
}

// Add tracking structs for complete lookup (optional - for advanced tracking)
type CompleteLookupResponse struct {
    QueryEmbed   []float64                `json:"query_embed"`
    TargetNodeID []byte                   `json:"target_node_id"`
    LookupPath   []LookupStep             `json:"lookup_path"`
    FinalResult  *EmbeddingSearchResponse `json:"final_result"`
    TotalHops    int                      `json:"total_hops"`
    Success      bool                     `json:"success"`
    StartTime    int64                    `json:"start_time"`
    EndTime      int64                    `json:"end_time"`
}

type LookupStep struct {
    NodeID           []byte     `json:"node_id"`
    PeerID           string     `json:"peer_id"`
    StepNumber       int        `json:"step_number"`
    NodesReturned    []PeerInfo `json:"nodes_returned"`
    DistanceToTarget string     `json:"distance_to_target"`
    Timestamp        int64      `json:"timestamp"`
    ResponseTime     int64      `json:"response_time_ms"`
}
