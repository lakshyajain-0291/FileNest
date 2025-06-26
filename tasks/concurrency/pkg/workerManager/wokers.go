package workermanager

import (
	"conc-task/pkg/config"
	"conc-task/pkg/embeds"
	"conc-task/pkg/models"
	"fmt"
	"log"
	"os"
	"sync"
)


func Worker(jobChan chan models.FileJob, wg *sync.WaitGroup, WorkerIndex int) {
    for {
        job, ok := <-jobChan
		//The !ok part simply handles the case where jobChan has been closed. 
        if !ok {
            return
        }

		// processChan is used to check if processJob has completed or not.
		processChan := make(chan bool)
		var similarity float64
		// processJob is a goroutine since in the code after, we are waiting on both processJob and context timeout.
		go func() {
			done, sim := processJob(&job)
			similarity = sim
			processChan <- done
		}()

		/*  Here, select waits on multiple channels and selects the one which is done first. 
			So, if timeout occurs first, wg.Done() is called and timeout is shown.
			If processChan is true first, then wg.Done() is called as job is done*/
        select {
        case <-job.Ctx.Done():
            fmt.Printf("Job for %s timed out or cancelled with err: %s\n", job.FileIndex.Filename, job.Ctx.Err())
            wg.Done()
        case <- processChan:
			fmt.Printf("[Worker-%v] Processed file: %v → D1TV: %v (similarity: %.2f)\n",
			WorkerIndex,job.FileIndex.Filename, job.FileIndex.D1tvID, similarity)
            wg.Done()
        }
    }
}

func processJob(job *models.FileJob) (bool, float64){
fileContent, err := os.ReadFile(job.FileIndex.Filepath)
	if err != nil {
		log.Printf("Error on reading file content: %v", err.Error())
		return false, 0
	}

	embedding := embeds.GenerateEmbed(string(fileContent))
	job.FileIndex.Embedding = embedding

	similarity := embeds.AssignCluster(&job.FileIndex, job.D1TVS)

	config.UpsertFile(job.DB, job.FileIndex)
	return true, similarity
}