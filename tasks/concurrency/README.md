# FileNest Concurrency Task (`tasks/concurrency/`)

This folder implements a concurrent file indexing and embedding system for the FileNest project. It uses Go's goroutines and channels to efficiently process and cluster file embeddings, storing results in a PostgreSQL database via GORM.

---

## 📁 Folder Structure

```
tasks/
└── concurrency/
    ├── go.mod, go.sum                # Go module files
    ├── cmd/
    │   └── main.go                   # Entry point: sets up workers and jobs
    ├── pkg/
    │   ├── config/
    │   │   └── app.go                # DB connection and upsert logic
    │   ├── embeds/
    │   │   └── embedding.go          # Embedding generation and similarity logic
    │   ├── models/
    │   │   ├── fileModel.go          # FileIndex struct and file discovery
    │   │   └── workerModels.go       # FileJob struct for worker jobs
    │   └── workerManager/
    │       └── wokers.go             # Worker pool and job processing logic
    └── sample_embeds/
        ├── e1.txt, e2.txt, ...       # Sample files for embedding
```

---

## 🚀 Quick Start

### Prerequisites

- Go 1.21 or higher
- PostgreSQL running locally (`user: postgres`, `password: password`, `database: filenest`)
- The `file_index` table created in your database (see [Database Schema](#database-schema))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AISocietyIITJ/FileNest.git
   cd FileNest/tasks/concurrency
   ```

2. **Install dependencies**
   ```bash
   go mod tidy
   ```

3. **Prepare the database**
   - Ensure PostgreSQL is running and accessible.
   - Create the `filenest` database and the `file_index` table (see below).

4. **Run the program**
   ```bash
   go run cmd/main.go -w 5 -dir ./sample_embeds
   ```
   - `-w`: Number of concurrent workers (default: 5)
   - `-dir`: Directory containing files to index (default: `./sample_embeds`)

---

## 🗄️ Database Schema

The `file_index` table should look like:

```sql
CREATE TABLE file_index (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    filepath TEXT,
    embedding FLOAT8[],
    d1tv_id INT,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🧩 Code Structure & Developer Documentation

### 1. Main Program

- **[`cmd/main.go`](cmd/main.go)**
  - Parses flags for worker count and directory path.
  - Initializes the database connection.
  - Generates random cluster embeddings (D1TVs).
  - Calls `processEmbeddings` to start the worker pool and distribute jobs.

### 2. Models

- **[`pkg/models/fileModel.go`](pkg/models/fileModel.go)**
  - `FileIndex`: Struct representing a file's metadata and embedding.
  - `GenerateFileIndices(dirPath string)`: Scans a directory and returns a slice of `FileIndex` for each file.

- **[`pkg/models/workerModels.go`](pkg/models/workerModels.go)**
  - `FileJob`: Struct containing all data needed for a worker to process a file (DB, embeddings, file info, context).

### 3. Embedding Logic

- **[`pkg/embeds/embedding.go`](pkg/embeds/embedding.go)**
  - `GenerateEmbed(content string)`: Generates a normalized random embedding vector for file content.
  - `CalcSimilarity(vec1, vec2)`: Computes cosine similarity between two vectors.
  - `AssignCluster(file *FileIndex, D1TVS [][]float64)`: Assigns the file to the most similar cluster and returns the similarity score.

### 4. Database Logic

- **[`pkg/config/app.go`](pkg/config/app.go)**
  - `InitDB()`: Connects to PostgreSQL using GORM.
  - `UpsertFile(db, file)`: Upserts a `FileIndex` record using the `id` as the unique key.

### 5. Worker Pool

- **[`pkg/workerManager/wokers.go`](pkg/workerManager/wokers.go)**
  - `Worker(jobChan, wg, WorkerIndex)`: Goroutine that processes jobs from the channel, handling timeouts and reporting results.
  - `processJob(job *FileJob)`: Reads file content, generates embedding, assigns cluster, and upserts the result.

---

## ⚙️ How It Works

1. **File Discovery:**  
   All files in the specified directory are discovered and wrapped as `FileIndex` structs.

2. **Embedding Generation:**  
   For each file, a random embedding vector is generated (simulating a real embedding model).

3. **Clustering:**  
   Each file's embedding is compared to a set of cluster vectors (D1TVs) using cosine similarity, and the file is assigned to the closest cluster.

4. **Concurrent Processing:**  
   A pool of worker goroutines process files in parallel, each with a timeout context to avoid hanging.

5. **Database Upsert:**  
   Each processed file's metadata, embedding, and cluster assignment are upserted into the PostgreSQL database.

---

## 🧪 Testing

- Place sample files in the `sample_embeds/` directory.
- Run the program and observe output like:
  ```
  [Worker-2] Processed file: e1.txt → D1TV: 3 (similarity: 0.87)
  ```
- Check your database for new/updated records in `file_index`.

---

## 📝 Extending & Customizing

- **Embedding Model:**  
  Replace `GenerateEmbed` with your own embedding logic for real file content.

- **Cluster Initialization:**  
  Adjust the number and type of D1TVs as needed for your use case.

- **Database Fields:**  
  Add fields to `FileIndex` and update the schema and upsert logic accordingly.

- **Error Handling:**  
  Improve error handling and logging for production use.

---

## 🛠️ Developer Tips

- Use the `-w` flag to tune concurrency for your system.
- Use context timeouts to prevent long-running jobs from blocking the worker pool.
- The worker pool pattern here is suitable for any batch file processing task.

---

## 📚 References

- [GORM Documentation](https://gorm.io/docs/)
- [Go Concurrency Patterns](https://go.dev/doc/effective_go#concurrency)
- [PostgreSQL Arrays](https://www.postgresql.org/docs/current/arrays.html)

---

## 🛡️ License

This project is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.

---