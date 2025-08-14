package storage

import (
	"kademlia/pkg/embedding"
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

func (s *SQLiteStorage) StoreNodeEmbedding(nodeID []byte, embeddingVec []float64) error {
	nodeEmbedding := NodeEmbedding{
		NodeID:    nodeID,
		Embedding: EmbeddingVector(embeddingVec),
	}

	// Use GORM's Upsert (Create or Update)
	result := s.db.Where("node_id = ?", nodeID).Assign(nodeEmbedding).FirstOrCreate(&nodeEmbedding)
	return result.Error
}

func (s *SQLiteStorage) FindClosestNodes(queryEmbed []float64, limit int) ([]EmbeddingResult, error) {
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

		results = append(results, EmbeddingResult{
			Key:     ne.NodeID,
			Embedding:  []float64(ne.Embedding),
			Similarity: similarity,
		})
	}

	// Sort by similarity (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Limit results
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
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
        if similarity >= threshold {
            results = append(results, EmbeddingResult{
                Key:     ne.NodeID,
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
    
    return results, nil
}

func (s *SQLiteStorage) Close() error {
	sqlDB, err := s.db.DB()
	if err != nil {
		return err
	}
	return sqlDB.Close()
}
