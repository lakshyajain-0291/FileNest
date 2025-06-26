package indexer

import (
	"context"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/centauri1219/FileNest/tasks/prototype/internal/database"
	"github.com/centauri1219/FileNest/tasks/prototype/internal/models"
)

type Indexer struct { //holds the database, d1tv centroids, worker etc
	db           *database.Database
	d1tvs        [][]float64
	workerCount  int
	timeout      time.Duration
	embeddingDim int
}

func NewIndexer(db *database.Database, workerCount int, timeoutSeconds int, embeddingDim int) *Indexer {
	return &Indexer{
		db:           db,
		d1tvs:        initializeD1TVs(embeddingDim),
		workerCount:  workerCount,
		timeout:      time.Duration(timeoutSeconds) * time.Second,
		embeddingDim: embeddingDim,
	}
}

func (idx *Indexer) IndexDirectory(ctx context.Context, dirPath string) error {
	// Create channels
	jobChan := make(chan models.FileJob, idx.workerCount*2)          // holds file path and content to be processed
	resultChan := make(chan models.ProcessResult, idx.workerCount*2) // holds processed results like embedding

	// Start workers
	var wg sync.WaitGroup //tracks when all workers finish
	for i := 0; i < idx.workerCount; i++ {
		wg.Add(1) // add worker to the wait group
		go idx.worker(ctx, i+1, jobChan, resultChan, &wg)
	}

	// Start result processor
	var processorWg sync.WaitGroup
	processorWg.Add(1)
	go idx.resultProcessor(ctx, resultChan, &processorWg) //Reads from resultChan and inserts into PostgreSQL

	// Send jobs
	jobCount := 0
	err := idx.walkDirectory(ctx, dirPath, jobChan, &jobCount) //sends file jobs to jobChan
	close(jobChan)                                             // close job channel when done

	if err != nil {
		log.Printf("Error walking directory: %v", err)
	}

	// Wait for workers to finish
	wg.Wait()
	close(resultChan)

	// Wait for result processor to finish
	processorWg.Wait()

	log.Printf("Indexing completed. Processed %d files.", jobCount)
	return idx.printStats(ctx)
}

func (idx *Indexer) walkDirectory(ctx context.Context, dirPath string, jobChan chan<- models.FileJob, jobCount *int) error {
	supportedExts := map[string]bool{
		".txt": true, ".md": true, ".json": true,
		".py": true, ".go": true, ".js": true, ".html": true,
	}

	return filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error { //walks the directory tree
		if err != nil {
			log.Printf("Error accessing path %s: %v", path, err)
			return nil // Continue walking
		}

		if d.IsDir() { // Skip directories
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))
		if !supportedExts[ext] {
			return nil
		} // if the file extension is not supported, skip it

		// Check context cancellation
		select {
		case <-ctx.Done(): // if the context is cancelled, return the error
			return ctx.Err()
		default:
		}

		content, err := os.ReadFile(path)
		if err != nil {
			log.Printf("Error reading file %s: %v", path, err)
			return nil
		}

		// Skip empty files or files that are too large
		if len(content) == 0 || len(content) > 1024*1024 { // 1MB limit
			return nil
		}

		job := models.FileJob{
			Filepath: path,
			Content:  string(content), //creates job
		}

		select {
		case jobChan <- job: //sends the job to the job channel
			*jobCount++
		case <-ctx.Done():
			return ctx.Err()
		}

		return nil
	})
}

func (idx *Indexer) worker(ctx context.Context, workerID int, jobChan <-chan models.FileJob, resultChan chan<- models.ProcessResult, wg *sync.WaitGroup) { //go routine fucnction for each worker
	defer wg.Done()

	log.Printf("[Worker-%d] Started", workerID)
	defer log.Printf("[Worker-%d] Stopped", workerID)

	for {
		select {
		case job, ok := <-jobChan:
			if !ok {
				return // Channel closed
			}

			result := idx.processFile(ctx, workerID, job)

			select {
			case resultChan <- result:
			case <-ctx.Done():
				return
			}

		case <-ctx.Done():
			return
		}
	}
}

func (idx *Indexer) processFile(ctx context.Context, workerID int, job models.FileJob) models.ProcessResult {
	// Create timeout context for this file
	fileCtx, cancel := context.WithTimeout(ctx, idx.timeout) //timeout for processing
	defer cancel()

	result := models.ProcessResult{
		FileJob: job,
	}

	// Check for cancellation
	select {
	case <-fileCtx.Done():
		result.Error = fileCtx.Err()
		return result
	default:
	} // checks if content was cancelled before starting

	// Generate embedding
	embedding := generateEmbedding(job.Content, idx.embeddingDim)
	result.Embedding = embedding

	// Find best D1TV
	d1tvID, similarity := findBestD1TV(embedding, idx.d1tvs)
	result.D1TVID = d1tvID
	result.Similarity = similarity

	log.Printf("[Worker-%d] Processed file: %s , D1TV: %d (similarity: %.3f)",
		workerID, filepath.Base(job.Filepath), d1tvID, similarity)

	return result
}

func (idx *Indexer) resultProcessor(ctx context.Context, resultChan <-chan models.ProcessResult, wg *sync.WaitGroup) { // this go routine processes results from all the workers and stores them in the database
	defer wg.Done()

	log.Println("[ResultProcessor] Started")
	defer log.Println("[ResultProcessor] Stopped")

	for {
		select {
		case result, ok := <-resultChan:
			if !ok {
				return // Channel closed
			}

			if result.Error != nil {
				log.Printf("[ResultProcessor] Error processing %s: %v", result.FileJob.Filepath, result.Error)
				continue
			}

			fileIndex := &models.FileIndex{
				Filename:  filepath.Base(result.FileJob.Filepath),
				Filepath:  result.FileJob.Filepath,
				Embedding: result.Embedding,
				D1TVID:    result.D1TVID,
			}

			if err := idx.db.InsertFileIndex(ctx, fileIndex); err != nil { //inserts into db
				log.Printf("[ResultProcessor] Error storing %s: %v", result.FileJob.Filepath, err)
			}

		case <-ctx.Done():
			return
		}
	}
}

func (idx *Indexer) printStats(ctx context.Context) error {
	stats, err := idx.db.GetFileIndexStats(ctx) // map of d1tv_id to count of indexed files
	if err != nil {
		return err
	}

	log.Println("\n=== Indexing Statistics ===")
	totalFiles := 0
	for d1tvID, count := range stats {
		log.Printf("D1TV %d: %d files", d1tvID, count)
		totalFiles += count
	}
	log.Printf("Total indexed files: %d", totalFiles)
	log.Println("===========================")

	return nil
}
