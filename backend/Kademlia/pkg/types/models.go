package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

type NodeEmbedding struct {
    NodeID    []byte          `gorm:"column:node_id;primaryKey" json:"node_id"`
    Embedding EmbeddingVector `gorm:"column:embedding;type:text;not null" json:"embedding"`
}

// Custom type for storing float64 slice as JSON in database
type EmbeddingVector []float64

// Implement driver.Valuer interface for storing in database
func (e EmbeddingVector) Value() (driver.Value, error) {
    return json.Marshal(e)
}

// Implement sql.Scanner interface for reading from database
func (e *EmbeddingVector) Scan(value any) error {
    if value == nil {
        *e = nil
        return nil
    }
    
    bytes, ok := value.([]byte)
    if !ok {
        return fmt.Errorf("cannot scan %T into EmbeddingVector", value)
    }
    
    return json.Unmarshal(bytes, e)
}

// Table name
func (NodeEmbedding) TableName() string {
    return "node_embeddings"
}
