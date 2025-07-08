package model

// EmbeddingSearchRequest represents a request to search for similar peer embeddings
type EmbeddingSearchRequest struct {
	Source       string    `json:"source"`
	SourceID     int       `json:"source_id"`
	Embed        []float64 `json:"embed"`
	PrevDepth    int       `json:"prev_depth"`
	QueryType    string    `json:"query_type"`
	Threshold    float64   `json:"threshold"`
	ResultsCount int       `json:"results_count"`
}

// EmbeddingSearchResponse represents the response to an embedding search
type EmbeddingSearchResponse struct {
	Type          string    `json:"type"`
	QueryEmbed    []float64 `json:"query_embed"`
	Depth         int       `json:"depth"`
	CurrentPeerID int       `json:"current_peer_id"`
	NextPeerID    int       `json:"next_peer_id"`
	IsProcessed   bool      `json:"is_processed"`
}

// type Files struct {
// 	ID          uuid.UUID `json:"id" db:"id"`
// 	FileHash    string    `json:"file_hash" db:"file_hash"`
// 	Filename    string    `json:"filename" db:"filename"`
// 	FileSize    int64     `json:"file_size" db:"file_size"`
// 	ContentType string    `json:"content_type" db:"content_type"`
// 	CreatedAt   time.Time `json:"created_at" db:"created_at"`
// }

// type Peer struct {
// 	ID        uuid.UUID `json:"id" db:"id"`
// 	PeerID    string    `json:"peer_id" db:"peer_id"`
// 	Name      string    `json:"name" db:"name"`
// 	IPAddress string    `json:"ip_address" db:"ip_address"`
// 	IsOnline  bool      `json:"is_online" db:"is_online"`
// 	LastSeen  time.Time `json:"last_seen" db:"last_seen"`
// 	CreatedAt time.Time `json:"created_at" db:"created_at"`
// }

// type PeerFile struct {
// 	ID          uuid.UUID `json:"id" db:"id"`
// 	PeerID      uuid.UUID `json:"peer_id" db:"peer_id"`
// 	FileID      uuid.UUID `json:"file_id" db:"file_id"`
// 	AnnouncedAt time.Time `json:"announced_at" db:"announced_at"`
// }

// type TrustScore struct {
// 	ID                  uuid.UUID `json:"id" db:"id"`
// 	PeerID              uuid.UUID `json:"peer_id" db:"peer_id"`
// 	Score               float64   `json:"score" db:"score"`
// 	SuccessfulTransfers int       `json:"successful_transfers" db:"successful_transfers"`
// 	FailedTransfers     int       `json:"failed_transfers" db:"failed_transfers"`
// 	UpdatedAt           time.Time `json:"updated_at" db:"updated_at"`
// }

// type ActiveConnection struct {
// 	ID          uuid.UUID  `json:"id" db:"id"`
// 	RequesterID uuid.UUID  `json:"requester_id" db:"requester_id"`
// 	ProviderID  uuid.UUID  `json:"provider_id" db:"provider_id"`
// 	FileID      uuid.UUID  `json:"file_id" db:"file_id"`
// 	Status      string     `json:"status" db:"status"`
// 	StartedAt   time.Time  `json:"started_at" db:"started_at"`
// 	CompletedAt *time.Time `json:"completed_at" db:"completed_at"`
//}
