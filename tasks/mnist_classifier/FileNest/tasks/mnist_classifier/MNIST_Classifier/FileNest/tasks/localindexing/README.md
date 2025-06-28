# FileNest - Local Indexing Phase

A minimal, concurrent file embedding and D1TV assignment service in Go. Indexes files from a directory, generates semantic embeddings, assigns each file to a random D1TV centroid, and stores metadata and embeddings in PostgreSQL.

---

## 🚀 Features

- **Concurrent File Processing:** Multi-worker pipeline for fast indexing.
- **Semantic Embeddings:** Generates 128-dim float embeddings (random for demo).
- **D1TV Assignment:** Assigns files to the nearest centroid using cosine similarity.
- **PostgreSQL Storage:** Stores file metadata, embeddings, and D1TV assignments.
- **Graceful Shutdown:** Handles interrupts and cleans up resources.

---

## 🏗️ Project Structure

```
filenest-xs/
├── main.go               # Entry point, worker pool, orchestration
├── sample_files/         # Example files to index
│   ├── text.txt, data.json, readme.md
├── go.mod, go.sum        # Go dependencies
├── .env                  # (Optional) Environment variables
```

---

## ⚙️ Setup & Usage

### 1. Prerequisites

- Go 1.23+
- PostgreSQL (default: user `postgres`, password `postgres`, db `filenest_xs`)
- (Optional) Directory of sample files

### 2. Database Setup

Create the database:

```sh
createdb -h localhost -U postgres filenest_xs
```

### 3. Install Dependencies

```sh
go mod tidy
```

### 4. Run the Indexer

```sh
go run main.go -dir=./sample_files -workers=5
```

- `-dir`: Directory containing files (default: `./sample_files`)
- `-workers`: Number of concurrent workers (default: 5)

---

## 🗃️ Database Schema

Table: `file_index`

| Column    | Type         | Description                        |
|-----------|--------------|------------------------------------|
| id        | SERIAL       | Primary key                        |
| file_name | TEXT         | File name (unique with path)       |
| file_path | TEXT         | Full file path (unique with name)  |
| embedding | float8[]     | 128-dim embedding vector           |
| d1tv_id   | INTEGER      | Assigned D1TV centroid             |
| indexed_at| TIMESTAMP    | Time of indexing                   |

---

## 🧩 How It Works

1. **main.go**: Reads files, spawns workers, orchestrates the pipeline.
2. **d1tv/d1tv.go**: Generates random embeddings, assigns to nearest centroid.
3. **database/db.go**: Handles DB connection, migration, and upsert.
4. **model/model.go**: GORM model for file metadata and embeddings.

---

## 📝 License

MIT License. See [LICENSE](../../LICENSE) for details.

---

## 🙏 Acknowledgments

Inspired by the FileNest project (Summer RAID 2025, IIT Jodhpur).
