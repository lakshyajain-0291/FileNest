# FileNest CRUD API (`tasks/crud/`)

This folder contains a simple RESTful CRUD API for managing file metadata in the FileNest project. It is written in Go using the [Gin](https://github.com/gin-gonic/gin) web framework and MongoDB as the backend database.

---

## 📂 Folder Structure

```
tasks/
└── crud/
    ├── .env                  # Environment variables (MongoDB URI, etc.)
    ├── main.go               # Entry point for the CRUD API server
    ├── go.mod, go.sum        # Go module files
    └── pkg/
        ├── config/
        │   └── app.go        # Database connection logic
        ├── controllers/
        │   └── fileController.go  # HTTP handlers for CRUD operations
        ├── models/
        │   └── models.go     # File metadata struct
        └── utils/
            └── utils.go      # Utility functions (error handling)
```

---

## 🚀 Quick Start

### Prerequisites

- Go 1.21 or higher
- MongoDB instance (local or remote)

### Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AISocietyIITJ/FileNest.git
   cd FileNest/tasks/crud
   ```

2. **Configure Environment Variables**  
   Create a `.env` file in `tasks/crud/` with:
   ```
   MONGO_URI=mongodb://localhost:27017
   or
   Use a MongoDB Cluster URI.
   ```

3. **Install dependencies**  
   ```bash
   go mod tidy
   ```

4. **Run the server**  
   ```bash
   go run main.go
   ```
   The API will be available at `http://localhost:8000`.

---

## 🛠️ API Endpoints

| Method | Endpoint            | Description                  |
|--------|---------------------|------------------------------|
| GET    | `/files`            | List all files               |
| GET    | `/files/:id`        | Get file by ID               |
| POST   | `/files`            | Create a new file metadata   |
| PUT    | `/files/:id`        | Update file metadata by ID   |
| DELETE | `/files/:id`        | Delete file metadata by ID   |

### Example: File Metadata JSON

```json
{
  "id": 1,
  "filename": "report.pdf",
  "filepath": "/docs/report.pdf",
  "filesize": 123456,
  "content_type": "application/pdf",
  "created_at": "2024-06-26T12:00:00Z",
  "updated_at": "2024-06-26T12:00:00Z"
}
```

---

## 🧩 Code Structure

### 1. Models

- [`models.go`](pkg/models/models.go):  
  Defines the `FileMetadata` struct, which represents the schema for file metadata stored in MongoDB.

### 2. Database Configuration

- [`app.go`](pkg/config/app.go):  
  - Loads the MongoDB URI from the environment.
  - Initializes and pings the MongoDB client.
  - Provides a helper to fetch a collection handle.

### 3. Controllers

- [`fileController.go`](pkg/controllers/fileController.go):  
  - Implements all CRUD HTTP handlers.
  - Uses Gin context for request/response.
  - Handles errors using utility functions.

### 4. Utilities

- [`utils.go`](pkg/utils/utils.go):  
  - Provides `ThrowError` for consistent error responses.

---

## ⚙️ Developer Notes

### Environment Variables

- The API expects `MONGO_URI` to be set in `.env`.
- Use [github.com/joho/godotenv](https://github.com/joho/godotenv) in `main.go` to load `.env` automatically.

### MongoDB Collections

- The API uses two databases:  
  - `filesCluster` (for storing file metadata in the `Files` collection)

### Error Handling

- All errors are returned as JSON with a consistent structure:
  ```json
  {
    "Error: ": "error message",
    "Context: ": "contextual info"
  }
  ```

### Adding New Fields

- To add new metadata fields, update the `FileMetadata` struct in [`models.go`](pkg/models/models.go) and ensure the controller logic handles them.

---

## 🧪 Testing

You can use [curl](https://curl.se/) or [Postman](https://www.postman.com/) to test the endpoints.  
Example:

```bash
curl -X POST http://localhost:8000/files \
  -H "Content-Type: application/json" \
  -d '{"id":2,"filename":"test.txt","filepath":"/tmp/test.txt","filesize":100,"content_type":"text/plain"}'
```

---

## 🛡️ License

This project is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.

---