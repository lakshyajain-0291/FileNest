package usecase

import (
	"database/sql"
	"encoding/json"

	"github.com/centauri1219/Filenest/backend/p2p-api/tempcode/repository"
)

type FileService struct {
	DB *sql.DB
}

type Response struct {
	Data  interface{} `json:"data,omitempty"`
	Error string      `json:"error,omitempty"`
}

type WSRequest struct {
	Action  string          `json:"action"`
	Payload json.RawMessage `json:"payload"`
}

type WSResponse struct {
	Data  interface{} `json:"data,omitempty"`
	Error string      `json:"error,omitempty"`
}

func (svc *FileService) HandleWSMessage(msg []byte) ([]byte, error) {
	var searchReq struct {
		Source       string    `json:"source"`
		SourceID     int       `json:"source_id"`
		Embed        []float64 `json:"embed"`
		PrevDepth    int       `json:"prev_depth"`
		QueryType    string    `json:"query_type"`
		Threshold    float64   `json:"threshold"`
		ResultsCount int       `json:"results_count"`
	}
	if err := json.Unmarshal(msg, &searchReq); err != nil || searchReq.QueryType != "search" {
		return json.Marshal(map[string]interface{}{"error": "invalid request format or query_type"})
	}
	// Perform similarity search
	nextRepo := repository.NextPeersRepo{DB: svc.DB}
	peers, err := nextRepo.GetAllNextPeers()
	if err != nil {
		return json.Marshal(map[string]interface{}{"error": "db error: " + err.Error()})
	}
	// Compute cosine similarity for each peer
	bestSim := -1.0
	bestPeerID := 0
	for _, peer := range peers {
		sim := cosineSimilarity(searchReq.Embed, peer.Embedding)
		if sim > bestSim && sim >= searchReq.Threshold {
			bestSim = sim
			bestPeerID = peer.PeerID
		}
	}
	// Build response
	resp := map[string]interface{}{
		"type":            "search_result",
		"query_embed":     searchReq.Embed,
		"depth":           searchReq.PrevDepth,
		"current_peer_id": searchReq.SourceID,
		"next_peer_id":    bestPeerID,
		"is_processed":    bestSim >= 0,
	}
	return json.Marshal(resp)
}

// cosineSimilarity computes the cosine similarity between two float64 slices
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return -1
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return -1
	}
	return dot / (sqrt(normA) * sqrt(normB))
}

func sqrt(x float64) float64 {
	// Use Newton's method for square root
	z := x
	for i := 0; i < 10; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
