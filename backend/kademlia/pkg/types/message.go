package types

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
