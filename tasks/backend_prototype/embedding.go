package main

import (
	"math"
	"math/rand"
)

const embeddingDim = 128

func generateEmbedding(content string) []float64 { //generateEmbedding creates a random embedding vector for the given content, whereas in rela application, we will use a pre-trained model to generate the embedding
	vec := make([]float64, embeddingDim)
	for i := range vec {
		vec[i] = rand.Float64()
	}
	return vec
}

func cosineSimilarity(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := 0; i < embeddingDim; i++ {
		dot += a[i] * b[i]   //sum of all dot products
		normA += a[i] * a[i] //sum of all square of first vector
		normB += b[i] * b[i]
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB)) //cosine similarity s
}
