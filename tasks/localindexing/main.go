package main

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"time"

	"github.com/lib/pq" // PostgreSQL driver
)

type FileJob struct {
	Path string
}

type D1TV struct {
	ID       int
	Centroid []float64 //128 dimensional vector
}

type Config struct {
	DirPath      string
	WorkerCount  int //5
	Timeout      time.Duration
	DBConnStr    string
	EmbeddingDim int //128
	D1TVCount    int //10
}

type Indexer struct {
	config Config
	d1tvs  []D1TV
	db     *sql.DB
	jobs   chan FileJob
	wg     sync.WaitGroup
	ctx    context.Context
	cancel context.CancelFunc
}

func generateRandomVector(dim int) []float64 {
	vec := make([]float64, dim)
	rand.Seed(time.Now().UnixNano())
	for i := range vec {
		vec[i] = rand.Float64()*2 - 1
	}

	return normalizeVector(vec)
}

func normalizeVector(vec []float64) []float64 {
	magnitude := 0.0
	for _, v := range vec {
		magnitude += v * v
	}
	magnitude = math.Sqrt(magnitude)
	if magnitude == 0 {
		return vec
	}
	result := make([]float64, len(vec))
	for i, v := range vec {
		result[i] = v / magnitude
	}
	return result
}

func generateEmbedding() []float64 {
	const dim = 128
	vec := make([]float64, dim)
	var norm float64

	for i := 0; i < dim; i++ {
		vec[i] = rand.Float64()
		norm += vec[i] * vec[i]
	}

	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] /= norm
	}

	return vec
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	dotProduct := 0.0
	normA, normB := 0.0, 0.0
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func initDB(connStr string) (*sql.DB, error) {
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %v", err)
	}

	query := `
		CREATE TABLE IF NOT EXISTS file_index (
			id SERIAL PRIMARY KEY,
			filename TEXT,
			filepath TEXT,
			embedding FLOAT8[],
			d1tv_id INT,
			indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`
	_, err = db.Exec(query)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create table: %v", err)
	}
	return db, nil
}

func NewIndexer(config Config) (*Indexer, error) {
	ctx, cancel := context.WithCancel(context.Background())
	db, err := initDB(config.DBConnStr)
	if err != nil {
		return nil, err
	}

	d1tvs := make([]D1TV, config.D1TVCount)
	for i := 0; i < config.D1TVCount; i++ {
		d1tvs[i] = D1TV{
			ID:       i + 1,
			Centroid: generateEmbedding(),
		}
	}

	return &Indexer{
		config: config,
		d1tvs:  d1tvs,
		db:     db,
		jobs:   make(chan FileJob, 100), // Buffered channel
		ctx:    ctx,
		cancel: cancel,
	}, nil
}

func (idx *Indexer) Start() error {
	for i := 0; i < idx.config.WorkerCount; i++ {
		idx.wg.Add(1)
		go idx.worker(i)
	}

	go idx.enqueueJobs()

	go idx.handleSignals()

	idx.wg.Wait()

	return idx.db.Close()
}

func (idx *Indexer) enqueueJobs() {
	defer close(idx.jobs) // Close channel when done
	err := filepath.Walk(idx.config.DirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && isSupportedFile(path) {
			select {
			case idx.jobs <- FileJob{Path: path}:
			case <-idx.ctx.Done():
				return idx.ctx.Err()
			}
		}
		return nil
	})
	if err != nil {
		fmt.Printf("Error walking directory: %v\n", err)
	}
}

func isSupportedFile(path string) bool {
	ext := filepath.Ext(path)
	return ext == ".txt" || ext == ".md" || ext == ".json"
}

func (idx *Indexer) worker(id int) {
	defer idx.wg.Done()
	for job := range idx.jobs {
		if err := idx.processFile(id, job); err != nil {
			fmt.Printf("[Worker-%d] Error processing %s: %v\n", id, job.Path, err)
		}
	}
}

func (idx *Indexer) processFile(workerID int, job FileJob) error {
	ctx, cancel := context.WithTimeout(idx.ctx, idx.config.Timeout)
	defer cancel()
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	embedding := generateEmbedding()

	d1tvID, similarity := idx.findClosestD1TV(embedding)

	filename := filepath.Base(job.Path)
	query := `
		INSERT INTO file_index (filename, filepath, embedding, d1tv_id)
		VALUES ($1, $2, $3, $4)`
	_, err := idx.db.ExecContext(ctx, query, filename, job.Path, pq.Array(embedding), d1tvID)
	if err != nil {
		return fmt.Errorf("failed to insert into database: %v", err)
	}
	// Log result
	fmt.Printf("[Worker-%d] Processed file: %s → D1TV: %d (similarity: %.2f)\n",
		workerID, filename, d1tvID, similarity)
	return nil
}

func (idx *Indexer) findClosestD1TV(embedding []float64) (int, float64) {
	maxSimilarity := -1.0
	var closestID int
	for _, d1tv := range idx.d1tvs {
		sim := cosineSimilarity(embedding, d1tv.Centroid)
		if sim > maxSimilarity {
			maxSimilarity = sim
			closestID = d1tv.ID
		}
	}
	return closestID, maxSimilarity
}

func (idx *Indexer) handleSignals() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt)
	<-sigChan
	fmt.Println("\nReceived interrupt signal, shutting down...")
	idx.cancel()
}

func main() {
	fmt.Print("Enter Directory Path: ")
	var dirPath string
	fmt.Scan(&dirPath)
	config := Config{
		DirPath:      dirPath,
		WorkerCount:  5,
		Timeout:      5 * time.Second,
		DBConnStr:    "host=localhost port=5432 user=postgres password=7291 dbname=filenest sslmode=disable",
		EmbeddingDim: 128,
		D1TVCount:    10,
	}

	rand.Seed(time.Now().UnixNano())

	indexer, err := NewIndexer(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create indexer: %v\n", err)
		os.Exit(1)
	}

	if err := indexer.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Indexing failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Indexing completed successfully")
}
