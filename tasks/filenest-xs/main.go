package main

import (
	"context"
	"filenest-xs/d1tv"
	"filenest-xs/database"
	"filenest-xs/model"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"

	"gorm.io/gorm"
)

// FileJob represents a file to be processed
type FileJob struct {
	FileName string
	FilePath string
}	

func main() {
	// Parse command-line flags
	dirPath := flag.String("dir", "./sample_files", "Directory of files to process")
	numWorkers := flag.Int("workers", 5, "Number of concurrent workers")
	flag.Parse()

	// Setup context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle interrupt signals (e.g., Ctrl+C)
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigs
		fmt.Println("\nShutting down...")
		cancel()
	}()

	// Initialize database
	db, err := database.DbInit()
	if err != nil {
		log.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		log.Fatal(err)
	}
	defer sqlDB.Close()

	// Read files from directory
	entries, err := os.ReadDir(*dirPath)
	if err != nil {
		log.Fatalf("Failed to read directory: %v", err)
	}

	// Start worker pool
	jobs := make(chan FileJob)
	var wg sync.WaitGroup
	for i := 0; i < *numWorkers; i++ {
		wg.Add(1)
		go startWorker(ctx, db, jobs, &wg, i+1) // Pass worker ID (1-based)
	}

	// Send jobs to workers
LoopToBreak:
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		select {
		case <-ctx.Done():
			break LoopToBreak
		case jobs <- FileJob{
			FileName: entry.Name(),
			FilePath: filepath.Join(*dirPath, entry.Name()),
		}:
		}
	}

	close(jobs)
	wg.Wait()

	if ctx.Err() != nil {
		fmt.Println("Processing interrupted before completion.")
	} else {
		fmt.Println("All files processed.")
	}
}

// startWorker processes jobs from the channel
func startWorker(ctx context.Context, db *gorm.DB, jobs <-chan FileJob, wg *sync.WaitGroup, workerID int) {
	defer wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case job, ok := <-jobs:
			if !ok {
				return
			}
			processFile(job, db, workerID)
		}
	}
}

// processFile reads, embeds, assigns, and saves file info
func processFile(job FileJob, db *gorm.DB, workerID int) {
	content, err := os.ReadFile(job.FilePath)
	if err != nil {
		log.Printf("[Worker-%d] Error reading %s: %v", workerID, job.FilePath, err)
		return
	}

	vec := d1tv.GenerateEmbeddings(string(content))
	treeID, similarity := d1tv.AssignToD1TV(vec)

	// Log in requested format
	fmt.Printf("[Worker-%d] Processed file: %s → D1TV: %d (similarity: %.2f)\n",
		workerID, job.FileName, treeID, similarity)

	fileIndex := &model.FileIndex{
		FileName:  job.FileName,
		FilePath:  job.FilePath,
		Embedding: vec,
		D1TVID:    treeID,
	}

	if err := database.UpsertFileIndex(db, fileIndex); err != nil {
		log.Printf("[Worker-%d] DB error for %s: %v", workerID, job.FileName, err)
	}
}
