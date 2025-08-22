package embedding

import (
    "math"
)

// CosineSimilarity calculates cosine similarity between two embeddings
// Returns a value between -1 and 1, where 1 means identical, 0 means orthogonal, -1 means opposite
func CosineSimilarity(a, b []float64) float64 {
    if len(a) != len(b) {
        return 0.0
    }
    
    if len(a) == 0 {
        return 0.0
    }
    
    var dotProduct, normA, normB float64
    
    for i := 0; i < len(a); i++ {
        dotProduct += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    
    // Avoid division by zero
    if normA == 0 || normB == 0 {
        return 0.0
    }
    
    return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// EuclideanDistance calculates Euclidean distance between two embeddings
// Lower values mean more similar embeddings
func EuclideanDistance(a, b []float64) float64 {
    if len(a) != len(b) {
        return math.MaxFloat64
    }
    
    if len(a) == 0 {
        return 0.0
    }
    
    var sum float64
    for i := 0; i < len(a); i++ {
        diff := a[i] - b[i]
        sum += diff * diff
    }
    
    return math.Sqrt(sum)
}

// ManhattanDistance calculates Manhattan (L1) distance between two embeddings
// Lower values mean more similar embeddings
func ManhattanDistance(a, b []float64) float64 {
    if len(a) != len(b) {
        return math.MaxFloat64
    }
    
    if len(a) == 0 {
        return 0.0
    }
    
    var sum float64
    for i := 0; i < len(a); i++ {
        sum += math.Abs(a[i] - b[i])
    }
    
    return sum
}

// DotProduct calculates dot product between two embeddings
func DotProduct(a, b []float64) float64 {
    if len(a) != len(b) {
        return 0.0
    }
    
    var sum float64
    for i := 0; i < len(a); i++ {
        sum += a[i] * b[i]
    }
    
    return sum
}

// IsSimilar checks if two embeddings are similar based on cosine similarity threshold
func IsSimilar(a, b []float64, threshold float64) bool {
    similarity := CosineSimilarity(a, b)
    return similarity >= threshold
}

// IsWithinDistance checks if two embeddings are within a certain Euclidean distance
func IsWithinDistance(a, b []float64, maxDistance float64) bool {
    distance := EuclideanDistance(a, b)
    return distance <= maxDistance
}

// Normalize normalizes an embedding vector to unit length
func Normalize(embedding []float64) []float64 {
    if len(embedding) == 0 {
        return embedding
    }
    
    var norm float64
    for _, val := range embedding {
        norm += val * val
    }
    norm = math.Sqrt(norm)
    
    if norm == 0 {
        return embedding
    }
    
    normalized := make([]float64, len(embedding))
    for i, val := range embedding {
        normalized[i] = val / norm
    }
    
    return normalized
}

// FindMostSimilar finds the most similar embedding from a slice of embeddings
func FindMostSimilar(query []float64, embeddings [][]float64) (int, float64) {
    if len(embeddings) == 0 {
        return -1, 0.0
    }
    
    bestIndex := 0
    bestSimilarity := CosineSimilarity(query, embeddings[0])
    
    for i := 1; i < len(embeddings); i++ {
        similarity := CosineSimilarity(query, embeddings[i])
        if similarity > bestSimilarity {
            bestSimilarity = similarity
            bestIndex = i
        }
    }
    
    return bestIndex, bestSimilarity
}

// FindClosest finds the closest embedding using Euclidean distance
func FindClosest(query []float64, embeddings [][]float64) (int, float64) {
    if len(embeddings) == 0 {
        return -1, math.MaxFloat64
    }
    
    bestIndex := 0
    bestDistance := EuclideanDistance(query, embeddings[0])
    
    for i := 1; i < len(embeddings); i++ {
        distance := EuclideanDistance(query, embeddings[i])
        if distance < bestDistance {
            bestDistance = distance
            bestIndex = i
        }
    }
    
    return bestIndex, bestDistance
}
