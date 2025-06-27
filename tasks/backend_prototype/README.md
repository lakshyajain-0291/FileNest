# FileNest Backend Prototype

This project is a backend prototype for indexing and storing file embeddings using Go, PostgreSQL (with GORM), and a worker pool architecture. It demonstrates concurrent file processing, embedding generation, and database storage.

## Features

- Walks a directory and processes all `.txt` files
- Generates random embeddings for each file (placeholder for real embedding models)
- Calculates cosine similarity to a set of reference embeddings (D1TV)
- Stores file metadata and embeddings in PostgreSQL using GORM
- Uses a worker pool for concurrent file processing
- Graceful shutdown on interrupt

## Project Structure

```
backend_prototype/
├── db/
│   └── database.go         # Database connection and initialization
├── embedding.go            # Embedding and similarity functions
├── go.mod                  # Go module definition
├── go.sum                  # Go dependencies
├── main.go                 # Application entry point
├── models/
│   └── model.go            # FileIndex model definition
├── process.go              # File processing logic
├── worker.go               # Worker pool and job processing
├── sample_texts/
│   ├── a.txt ... j.txt     # Sample text files for indexing
└── README.md               # Project documentation
```

## Prerequisites

- Go 1.21 or higher
- PostgreSQL database
- Set the `POSTGRES_DSN` environment variable with your PostgreSQL connection string

## Getting Started

1. **Install dependencies:**
    ```sh
    go mod tidy
    ```

2. **Set up your PostgreSQL DSN:**
    ```sh
    export POSTGRES_DSN="host=localhost user=youruser password=yourpass dbname=yourdb port=5432 sslmode=disable"
    ```

3. **Run the application:**
    ```sh
    go run main.go worker.go process.go embedding.go ./sample_texts
    ```

    By default, it will process all `.txt` files in `sample_texts/` using 5 workers.

4. **Command-line options:**
    - `-w`: Number of concurrent workers (default: 5)
    - `-dir`: Directory to index (default: ./sample_texts)
    - `-timeout`: Per-file processing timeout (default: 5s)

    Example:
    ```sh
    go run main.go worker.go process.go embedding.go db/database.go models/model.go -w 10 -dir ./sample_texts -timeout 10s
    ```

## How It Works

- The app walks through the specified directory, sending `.txt` files to a pool of workers.
- Each worker reads the file, generates a random embedding, finds the most similar D1TV embedding, and stores the result in PostgreSQL.
- Embeddings are stored as arrays using the `pq.Float64Array` type.

## Notes

- The embedding generation is a placeholder; replace `generateEmbedding` in `embedding.go` with your actual model.
- The database schema is auto-migrated on startup.
- Sample text files are provided in `sample_texts/`.