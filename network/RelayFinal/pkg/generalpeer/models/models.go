package models

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