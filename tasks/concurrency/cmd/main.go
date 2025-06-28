package main

import (
	"conc-task/pkg/config"
	"conc-task/pkg/embeds"
	"conc-task/pkg/models"
	workermanager "conc-task/pkg/workerManager"
	"context"
	"flag"
	"sync"
	"time"

	"gorm.io/gorm"
)

func main(){
	// This just takes maxWorker and dirPath as flags when doing go run
	maxWorkers := flag.Int("w", 5, "number of concurrent workers")
    dirPath := flag.String("dir", "./sample_embeds", "directory path to index")
    flag.Parse()
	
	var D1TV [][]float64 // stores the D1TVs of clusters to be used for similarity.
	db := config.InitDB()

	// Generate 10 D1TVs
	for range 10{
		D1TV = append(D1TV, embeds.GenerateEmbed(" "))
	}
	processEmbeddings(db, D1TV, *dirPath, maxWorkers)
}

func processEmbeddings(db *gorm.DB,D1TVs [][]float64, dirPath string, maxWorkers *int) {
	// jobChan is the channel which will be used by main routine to give jobs to worker pool.
	jobChan := make(chan models.FileJob)

	// This initializes 5 workers.
	ctx := context.Background()
	wg := sync.WaitGroup{}
	for i:=0;i<*maxWorkers;i++{
		go workermanager.Worker(jobChan, &wg, i)		
	}

	// This creates a valid arr of FileIndex
	fileIndices := models.GenerateFileIndices(dirPath)

	// here, we will put jobs into jobChan.
	for _,file := range fileIndices{
		jobCtx, cancel := context.WithTimeout(ctx, time.Second*5)
		job := models.FileJob{DB: db, D1TVS: D1TVs, FileIndex: file, Ctx: jobCtx, Cancel: cancel}
		wg.Add(1)
		//We send job to jobChan
		jobChan <- job
	}
	close(jobChan)
	wg.Wait()
}


