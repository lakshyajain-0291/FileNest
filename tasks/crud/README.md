# FileNest CRUD API

This is a simple RESTful API for managing file metadata, built in Go Language along with MongoDB. It provides endpoints to create, read, update, and delete file metadata records.

## Features

- Add new file metadata (Create)
- Retrieve all files or a file by ID (Read)
- Update file metadata by ID (Update)
- Delete file metadata by ID (Delete)
- MongoDB Atlas integration

## Folder Structure

```
tasks/crud/
├── controllers.go   # HTTP handlers for CRUD operations
├── db.go            # MongoDB connection and initialization
├── main.go          # Entry point, router setup
├── models.go        # File metadata model definition
├── go.mod           # Go module definition
├── go.sum           # Go dependencies
└── README.md        # This file
```

## Prerequisites

- Go 1.21 or higher
- MongoDB Atlas cluster (or local MongoDB instance)
- Set your MongoDB URI in `db.go` if needed

## Getting Started

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Abhinav-Kumar2/FileNest.git
    cd FileNest/tasks/crud
    ```

2. **Install dependencies:**
    ```sh
    go get go.mongodb.org/mongo-driver@latest
    go mod tidy
    ```

3. **Run the server:**
    ```sh
    go run main.go controllers.go db.go models.go
    ```

    The API will be available at `http://localhost:8000/api/files`.

## API Endpoints

| Method | Endpoint             | Description                |
|--------|----------------------|----------------------------|
| POST   | `/api/files`         | Create a new file metadata |
| GET    | `/api/files`         | Get all files              |
| GET    | `/api/files/{id}`    | Get file by ID             |
| PUT    | `/api/files/{id}`    | Update file by ID          |
| DELETE | `/api/files/{id}`    | Delete file by ID          |

### Example File Metadata JSON

```json
{
  "filename": "example.pdf",
  "filepath": "/files/example.pdf",
  "filesize": 123456,
  "content_type": "application/pdf"
}
```

## Notes

- The MongoDB connection string is hardcoded in [`db.go`](db.go). Update it to match your environment.
- The API uses the `fileNest` database and `files` collection by default.