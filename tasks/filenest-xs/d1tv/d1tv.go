package d1tv

import (
	"math"
	"math/rand"
)

const EmbeddingSize = 128
const NumCentroids = 10

// D1TVCentroids holds the randomly initialized centroids for D1TV assignment.
var D1TVCentroids [NumCentroids][]float64

// rng is a deterministic random number generator for reproducibility.
var rng = rand.New(rand.NewSource(42))

// init initializes the D1TV centroids with random vectors.
func init() {
	for i := 0; i < NumCentroids; i++ {
		vec := make([]float64, EmbeddingSize)
		for j := range vec {
			vec[j] = rng.Float64()
		}
		D1TVCentroids[i] = vec
	}
}

// GenerateEmbeddings creates a random normalized embedding vector for the input content.
func GenerateEmbeddings(content string) []float64 {
	const dim = 128
	vec := make([]float64, dim)
	var norm float64

	for i := 0; i < dim; i++ {
		vec[i] = rng.Float64()
		norm += vec[i] * vec[i]
	}

	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] /= norm
	}

	return vec
}

// CosineSimilarity computes the cosine similarity between two vectors.
func CosineSimilarity(a, b []float64) float64 {
	dot, normA, normB := 0.0, 0.0, 0.0
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// AssignToD1TV assigns the given vector to the closest centroid and returns its index and similarity.
func AssignToD1TV(vec []float64) (int, float64) {
	maxSim, assigned := -1.0, 0
	for i, centroid := range D1TVCentroids {
		sim := CosineSimilarity(vec, centroid)
		if sim > maxSim {
			maxSim = sim
			assigned = i
		}
	}
	return assigned, maxSim
}
