package models

type Peer struct {
	ID   int
	IP   string
	Port int
}

type MessageToPeer struct {
	SourcePeer   string    `json:"source"`
	SourcePeerID int       `json:"source_id"`
	QueryEmbed   []float64 `json:"embed"`
	PrevDepth    int       `json:"prev_depth"`
	QueryType    string    `json:"query_type"`
	Threshold    float64   `json:"threshold"`
	ResultsCount int       `json:"results_count"`
	TargetPeerID int       `json:"target_peer_id"` //the target peer id is required at all times to find the nearest nodes
}

type Message struct {
	Type          string       `json:"type"`
	QueryEmbed    []float64    `json:"query_embed"`
	Depth         int          `json:"depth"`
	CurrentPeerID int          `json:"current_peer_id"`
	NextPeerID    int          `json:"next_peer_id"`
	FileMetadata  FileMetadata `json:"file_metadata"`
	IsProcessed   bool         `json:"is_processed"` //this is to check if the query has the reached the D4 node or not
	Found         bool         `json:"found"`        //this is to confirm if the target peer id for a peer at any depth has been found or not
}

type FileMetadata struct {
	Name         string  `json:"name"`
	CreatedAt    string  `json:"created_at"`
	LastModified string  `json:"last_modified"`
	FileSize     float64 `json:"file_size"`
	UpdatedAt    string  `json:"updated_at"`
}

type D1TV struct {
	Vector []float64
	PeerID int
	Depth  int
}

type ClusterFile struct {
	Filename  string       `json:"filename"`
	Metadata  FileMetadata `json:"metadata"`
	Embedding []float64    `json:"embedding"`
}

type Cluster struct {
	Centroid []float64     `json:"centroid"`
	Files    []ClusterFile `json:"files"`
}

type ClusterWrapper struct {
	Clusters []Cluster `json:"Clusters"`
}

// ClusterRequest defines the expected JSON structure for the request body.
type ClusterRequest struct {
	Path string `json:"path"`
}

// ForwardQueryRequest defines the structure of the incoming JSON payload.
// It contains a list of peers and the query embedding to forward.
type ForwardQueryRequest struct {
	Peers []Peer    `json:"peers"` // List of target peers (ID, IP, Port)
	Query []float64 `json:"query"` // Query embedding to forward
}
