package db

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv" // ✅ imported
)

var DB *pgxpool.Pool

func InitDB() {
	// ✅ Load environment variables from .env
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	url := os.Getenv("DATABASE_URL")
	if url == "" {
		log.Fatal("DATABASE_URL not set in environment")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	DB, err = pgxpool.New(ctx, url)
	if err != nil {
		log.Fatal("Failed to connect to DB:", err)
	}
}

func InsertFile(ctx context.Context, filename, path string, embedding []float64, d1tvID int) error {
	_, err := DB.Exec(ctx,
		"INSERT INTO file_index (filename, filepath, embedding, d1tv_id) VALUES ($1, $2, $3, $4)",
		filename, path, embedding, d1tvID,
	)
	return err
}
