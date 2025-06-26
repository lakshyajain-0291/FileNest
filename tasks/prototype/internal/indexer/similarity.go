package indexer

import "math"

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dot, norma, normb float64

	for i := range a {
		dot += a[i] * b[i]
		norma += a[i] * a[i]
		normb += b[i] * b[i]
	}

	norma = math.Sqrt(norma)
	normb = math.Sqrt(normb)

	if norma == 0 || normb == 0 {
		return 0
	}

	return dot / (norma * normb)
}

// findBestD1TV finds the D1TV with highest cosine similarity to the given embedding
func findBestD1TV(embedding []float64, d1tvs [][]float64) (int, float64) { //each d1tv is []float64 * 10 so basically [][10] float64
	bestd1tvid := 0
	bestd1tvsimilarity := -1.0

	for i, d1tv := range d1tvs {
		similarity := cosineSimilarity(embedding, d1tv)
		if similarity > bestd1tvsimilarity {
			bestd1tvsimilarity = similarity
			bestd1tvid = i
		}
	}

	return bestd1tvid, bestd1tvsimilarity
}
