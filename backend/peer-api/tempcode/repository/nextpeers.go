package repository

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
)

type NextPeer struct {
	PeerID    int       `json:"peer_id" db:"peer_id"`
	Embedding []float64 `json:"embedding" db:"embedding"`
}

type NextPeersRepo struct {
	DB *sql.DB
}

// GetAllNextPeers fetches all peer_ids and embeddings from nextpeers table
func (r *NextPeersRepo) GetAllNextPeers() ([]NextPeer, error) {
	rows, err := r.DB.QueryContext(context.Background(), "SELECT peer_id, embedding FROM nextpeers")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var peers []NextPeer
	for rows.Next() {
		var peer NextPeer
		var embedStr string
		if err := rows.Scan(&peer.PeerID, &embedStr); err != nil {
			return nil, err
		}
		// Assume embedding is stored as JSON array string
		if err := json.Unmarshal([]byte(embedStr), &peer.Embedding); err != nil {
			return nil, fmt.Errorf("embedding decode error: %w", err)
		}
		peers = append(peers, peer)
	}
	return peers, nil
}
