package models

type Peer struct {
	
	ID string
	IP string
	Port int
}

type Message struct {
	Type string `json:"type"`
	QueryEmbed []float64 `json:"query_embed"`
	Depth int `json:"depth"`
	CurrentPeerID int `json:"current_peer_id"`
	NextPeerId int  `json:"next_peer_id"`
	FileMetadata FileMetadata `json:"file_metadata"`
	IsProcessed bool `json:"is_processed"`
}

type FileMetadata struct {
	Name      string `json:"name"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}
