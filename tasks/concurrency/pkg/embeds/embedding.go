package embeds

import (
	"conc-task/pkg/models"
	"math"
	"math/rand"
)

func GenerateEmbed(content string) []float64{
	var vec []float64
	var norm float64 = 0
	for range 128{
		randNo := rand.Float64()
		norm += randNo*randNo
		vec = append(vec, randNo)
	}
	for i := range vec {
		vec[i] = vec[i] / math.Sqrt(norm)
	}
	return vec	
}



func CalcSimilarity(vec1 []float64, vec2 []float64) float64 {
	var dot float64
	for i := range vec1 {
		dot += vec1[i] * vec2[i]
	}
	return dot
}

func AssignCluster(file *models.FileIndex, D1TVS [][]float64) float64 {
	var maxSimilarity float64 = 0
	var argmax int
	for j,taggingvec := range D1TVS{
		similarity := CalcSimilarity(file.Embedding,taggingvec)
		if(similarity > maxSimilarity){
			argmax = j
			maxSimilarity = similarity
		}
	}
	file.D1tvID = argmax
	return maxSimilarity
}