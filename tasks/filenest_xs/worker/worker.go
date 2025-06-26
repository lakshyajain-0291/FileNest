package worker

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/dkstlzk/FileNest_Fork/db"
	"github.com/dkstlzk/FileNest_Fork/embedding"
	"github.com/dkstlzk/FileNest_Fork/models"
	"github.com/dkstlzk/FileNest_Fork/similarity"
)

type FileJob struct {
	Filename string
	Path     string
}

func Worker(id int, jobs <-chan FileJob, wg *sync.WaitGroup, ctx context.Context) {
	defer wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case job, ok := <-jobs:
			if !ok {
				return
			}

			localCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			func() {
				defer cancel()

				content, err := os.ReadFile(job.Path)
				if err != nil {
					fmt.Println("[Worker", id, "] Failed:", job.Filename)
					return
				}

				embed := embedding.GenerateEmbedding(string(content))

				bestID := -1
				bestSim := -1.0
				for i, vec := range models.D1TVs {
					sim := similarity.CosineSimilarity(embed, vec)
					if sim > bestSim {
						bestSim = sim
						bestID = i
					}
				}

				err = db.InsertFile(localCtx, job.Filename, job.Path, embed, bestID)
				if err != nil {
					fmt.Println("[Worker", id, "] DB Error:", err)
				} else {
					fmt.Printf("[Worker-%d] Processed: %s → D1TV: %d (sim: %.2f)\n", id, job.Filename, bestID, bestSim)
				}
			}()
		}
	}
}
