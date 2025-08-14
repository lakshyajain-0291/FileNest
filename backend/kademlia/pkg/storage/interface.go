package storage

type EmbeddingResult struct {
	Key        []byte    `json:"key"`
	Embedding  []float64 `json:"embedding"`
	Similarity float64   `json:"similarity"`
}

type Interface interface {
	StoreNodeEmbedding(nodeID []byte, embeddingVec []float64) error
	FindClosestNodes(queryEmbed []float64, limit int) ([]EmbeddingResult, error)
	FindSimilar(queryEmbed []float64, threshold float64, limit int) ([]EmbeddingResult, error)
}
