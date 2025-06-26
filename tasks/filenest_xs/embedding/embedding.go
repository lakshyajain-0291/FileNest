package embedding

import (
	"math/rand"
)

func GenerateEmbedding(content string) []float64 {
	vec := make([]float64, 128)
	for i := range vec {
		vec[i] = rand.Float64()
	}
	return vec
}
