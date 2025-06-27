package main

import (
	"backend_prototype/db"
	"backend_prototype/models" // for importing the FileIndex struct
	"context"
	"log"
	"os" //for reading files
	"path/filepath"
	"time"
)

func processFile(ctx context.Context, workerID int, path string, d1tvs [][]float64) error { //mainly controllin cancellation and timeout of the worker and returning error if any
	select { //basic checking to see if the context is done, if so return the error
	case <-ctx.Done():
		return ctx.Err()
	default:
		content, err := os.ReadFile(path) //read the file as bytes
		if err != nil {
			return err
		}
		embedding := generateEmbedding(string(content)) //generate the embedding from the file content, this is a placeholder function

		bestID := 0
		bestSim := -1.0             //start with lowest similarity score
		for i, vec := range d1tvs { //looping through the D1TV embeddings to find the best match
			sim := cosineSimilarity(embedding, vec)
			if sim > bestSim {
				bestSim = sim
				bestID = i
			}
		}

		db.DB.Create(&models.FileIndex{ //creating fileindex in the databse and .models is here cuz we use package models in model.go
			Filename:  filepath.Base(path),
			Filepath:  path,
			Embedding: embedding,
			D1TVID:    bestID,
			IndexedAt: time.Now(),
		})

		log.Printf("[Worker-%d] Processed %s → D1TV: %d (similarity: %.2f)", workerID, filepath.Base(path), bestID, bestSim)
		return nil
	}
}
