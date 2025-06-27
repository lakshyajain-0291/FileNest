package main

import (
	"context"
	"log"
	"path/filepath"
	"time"
)

type FileJob struct {
	Path string
}

func startWorker(id int, jobs <-chan FileJob, d1tvs [][]float64, timeout time.Duration) { //worker function inside a goroutine, with paramters that are id, recieve channel of jobs, multi-dimensional slice, max time for worker to process a file
	for job := range jobs { //until all jobs are done
		ctx, cancel := context.WithTimeout(context.Background(), timeout) //context that starts from context.Background() and cancels after timeout duration
		err := processFile(ctx, id, job.Path, d1tvs)
		cancel() //don't use defer cancel since we don't want to lead to delayed cleanup
		if err != nil {
			log.Printf("[Worker-%d] Error processing %s: %v", id, filepath.Base(job.Path), err)
		}
	}
}
