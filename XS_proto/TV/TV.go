package TV

import (
    "math/rand"
    "sync"
    "time"
)

var (
    once sync.Once
    globalRand *rand.Rand
)

func initRand() {
    globalRand = rand.New(rand.NewSource(time.Now().UnixNano()))
}

func Generatetaggingvectors() [][]int {
    once.Do(initRand)
    
    taggingVectors := make([][]int, 10)
    for j := 0; j < 10; j++ {
        taggingVectors[j] = make([]int, 128)
        for i := 0; i < 128; i++ {
            taggingVectors[j][i] = globalRand.Intn(2)
        }
    }
    return taggingVectors
}
