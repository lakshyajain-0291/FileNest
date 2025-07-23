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
	Nodaught  int       `json:"no_daught" db:"no_daught"`
}

type NextPeersRepo struct {
	DB *sql.DB
}

// GetAllNextPeers fetches all peer_ids and embeddings from nextpeers table
func (r *NextPeersRepo) GetAllNextPeers() ([]NextPeer, error) {
	rows, err := r.DB.QueryContext(context.Background(), "SELECT peer_id, embedding, nextpeerdaught FROM nextpeers")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var peers []NextPeer
	for rows.Next() {
		var peer NextPeer
		var embedStr string
		if err := rows.Scan(&peer.PeerID, &embedStr, &peer.Nodaught); err != nil {
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

func (r *NextPeersRepo) UpdatePeerEmbedding(peerID int, newEmbedding []float64, newDaught int) error {
	jsonEmbed, err := json.Marshal(newEmbedding)
	if err != nil {
		return fmt.Errorf("embedding marshal error: %w", err)
	}

	_, err = r.DB.ExecContext(context.Background(),
		"UPDATE nextpeers SET embedding = $1, nextpeerdaught = $2 WHERE peer_id = $3",
		string(jsonEmbed), newDaught, peerID)
	return err
}

func (r *NextPeersRepo) InsertNewPeer(embed []float64) error {
	jsonEmbed, err := json.Marshal(embed)
	if err != nil {
		return fmt.Errorf("embedding marshal error: %w", err)
	}

	_, err = r.DB.ExecContext(context.Background(),
		`INSERT INTO nextpeers (embedding, nextpeerdaught) 
		 VALUES ($1, 0)`,
		string(jsonEmbed))
	return err
}
