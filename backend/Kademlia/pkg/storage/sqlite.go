package storage

import (
	"final/backend/pkg/embedding"
	"fmt"
	"log"
	"sort"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

type SQLiteStorage struct {
	db *gorm.DB
}

func NewSQLiteStorage(dbPath string) (*SQLiteStorage, error) {
	db, err := gorm.Open(sqlite.Open(dbPath), &gorm.Config{})
	if err != nil {
		return nil, err
	}

	// Auto migrate the schema - now using local NodeEmbedding struct
	err = db.AutoMigrate(&NodeEmbedding{})
	if err != nil {
		return nil, err
	}

	return &SQLiteStorage{
		db: db,
	}, nil
}

func (s *SQLiteStorage) FindSimilar(queryEmbed []float64, threshold float64, limit int) ([]EmbeddingResult, error) {
	var nodeEmbeddings []NodeEmbedding

	// Get all embeddings from database
	result := s.db.Find(&nodeEmbeddings)
	if result.Error != nil {
		return nil, result.Error
	}

	var results []EmbeddingResult

	// Calculate similarity for each stored embedding
	for _, ne := range nodeEmbeddings {
		similarity := embedding.CosineSimilarity(queryEmbed, []float64(ne.Embedding))

		// Only include embeddings that meet the threshold
		if similarity >= threshold{
			results = append(results, EmbeddingResult{
				Key:        ne.NodeID,
				Embedding:  []float64(ne.Embedding),
				Similarity: similarity,
			})
		}
	}

	// Sort by similarity (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Limit results
	if len(results) > limit {
		results = results[:limit]
	}
	log.Printf("results: %v", results)
	return results, nil
}

func (s *SQLiteStorage) Close() error {
	sqlDB, err := s.db.DB()
	if err != nil {
		return err
	}
	return sqlDB.Close()
}

func (s *SQLiteStorage) StoreNodeEmbedding(nodeID []byte, embeddingVec []float64) error {
    if len(nodeID) != 20 {
        return fmt.Errorf("nodeID must be 20 bytes (160 bits), got %d", len(nodeID))
    }

    nodeEmbedding := NodeEmbedding{
        NodeID:    nodeID,
        Embedding: EmbeddingVector(embeddingVec),
    }

    // Create a new record for each embedding. This allows multiple embeddings per NodeID.
    result := s.db.Create(&nodeEmbedding)
    return result.Error
}
