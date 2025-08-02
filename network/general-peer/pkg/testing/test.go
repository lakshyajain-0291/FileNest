package testing
func GenerateFakeVector(n int) []float64 {
	vec := make([]float64, n)
	for i := range n {
		vec[i] = float64(i) / float64(n)
	}
	return vec
}