# 📦 FileNest: Concurrent File Embedding & Clustering

A lightweight Go-based concurrent file processing system that reads text files from a directory, generates random embedding vectors for each, assigns them to the closest pre-generated D1TV cluster (based on cosine similarity), and saves the metadata into a PostgreSQL database.

Built using a worker pool pattern with goroutines, context-based graceful shutdown, and PostgreSQL insertion via pgx.

---

## 📂 Project Structure

.
├── main.go                → Main entry point, starts worker pool and processes jobs
├── db/
│   └── db.go              → Database connection management using pgx
├── models/
│   ├── d1tvs.go           → Pre-generated cluster vectors (D1TVs)
│   └── models.go          → File metadata struct
├── utils/
│   └── context.go         → Graceful shutdown context using signals
├── worker/
│   └── worker.go          → Worker pool logic and file processing routines
├── .env                   → Environment variables for DB connection
├── sample-data/           → Directory for sample text files
└── README.md              → README.md file

---

## 🚀 How to Run

### Prerequisites

- Go 1.21+
- PostgreSQL running locally
- A `.env` file with:

DATABASE_URL=postgres://username:password@localhost:5432/databasename?sslmode=disable

### Install dependencies

go mod tidy

### Run the program

go run main.go sample-data/

This will:
- Process all `.txt` files inside `sample-data/`
- Generate a 128-dimensional random embedding for each
- Assign each file to the nearest D1TV cluster based on cosine similarity
- Store metadata into PostgreSQL database

---

## 📊 Database Schema

CREATE TABLE files (
  id SERIAL PRIMARY KEY,
  filename TEXT,
  filepath TEXT,
  embedding FLOAT8[],
  d1tv_id INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

---

## ⚙️ How It Works

1. **Setup Context**
   - A cancellation context listens for system signals like `Ctrl+C` or termination signals for graceful shutdown.

2. **Database Connection**
   - PostgreSQL connection established via pgx, configured using `.env`.

3. **D1TV Initialization**
   - Pre-generate cluster vectors (D1TVs) for cosine similarity comparisons.

4. **Worker Pool Creation**
   - Spin up 5 worker goroutines via a channel-based worker pool.

5. **Job Distribution**
   - Walk through all `.txt` files inside the given directory, create a job for each, and send it to the worker pool.

6. **Embedding + Clustering**
   - Each worker reads file content, generates a 128-dim random embedding, finds the closest D1TV using cosine similarity, and inserts the metadata into the database.

7. **Wait for Completion**
   - The main routine waits for all worker goroutines to finish processing.

---

## 📚 Concepts Used

- Worker Pool pattern using goroutines + channels
- Context cancellation and graceful shutdown
- Embedding generation via random vectors
- Cosine similarity for clustering
- Concurrent PostgreSQL insertion with pgx
- Environment variable management via godotenv

---

## 📈 Possible Extensions

- Replace random embeddings with real embeddings from Sentence Transformers, CLIP, or other AI models.
- Add intelligent error handling, logging, and retry logic.
- Implement file deduplication using file content hash.
- Build a CLI interface for user-friendly file indexing.
- Optionally replace pgx with GORM for ORM-style database handling.

---

## 📝 License

MIT License
