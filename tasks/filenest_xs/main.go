package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/dkstlzk/FileNest_Fork/db"
	"github.com/dkstlzk/FileNest_Fork/models"
	"github.com/dkstlzk/FileNest_Fork/utils"
	"github.com/dkstlzk/FileNest_Fork/worker"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go ./sample-data")
		return
	}
	dir := os.Args[1]

	ctx := utils.SetupCancelContext()

	db.InitDB()
	defer db.DB.Close()

	models.InitD1TVs()

	jobs := make(chan worker.FileJob)
	var wg sync.WaitGroup

	for i := range [5]struct{}{} {
		wg.Add(1)
		go worker.Worker(i+1, jobs, &wg, ctx)
	}

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && (filepath.Ext(path) == ".txt") {
			jobs <- worker.FileJob{Filename: info.Name(), Path: path}
		}
		return nil
	})
	if err != nil {
		fmt.Println("Failed walking dir:", err)
	}

	close(jobs)
	wg.Wait()
	fmt.Println("All files processed.")
}
