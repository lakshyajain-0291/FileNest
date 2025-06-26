package indexer

import (
	"math"
	"math/rand"
)

func generateEmbedding(content string, dimension int) []float64 {
	var vec []float64 //empty slice to hold the embedding vector
	var norm float64 = 0
	for i := 0; i < dimension; i++ {
		randNo := rand.Float64()
		norm += randNo * randNo //sum of squares for normalization
		vec = append(vec, randNo)
	}
	for i := range vec {
		vec[i] = vec[i] / math.Sqrt(norm)
	}
	return vec
}

func normalizeVector(vec []float64) []float64 { //normalises vector to unit length
	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return vec
	}

	normalized := make([]float64, len(vec))
	for i, v := range vec {
		normalized[i] = v / norm
	}
	return normalized
}

// initializeD1TVs creates 10 predefined Depth 1 Tagging Vectors
func initializeD1TVs(dimension int) [][]float64 {
	d1tvs := make([][]float64, 10)

	// Create diverse, well-separated centroids
	seeds := []int64{
		1001, 2002, 3003, 4004, 5005,
		6006, 7007, 8008, 9009, 10010,
	} //10 unique seeds for random number generation

	for i, seed := range seeds {
		rng := rand.New(rand.NewSource(seed))
		d1tv := make([]float64, dimension) //create a new vector of the specified dimension

		for j := range d1tv {
			d1tv[j] = rng.NormFloat64()
		}

		d1tvs[i] = normalizeVector(d1tv)
	}

	return d1tvs
}
