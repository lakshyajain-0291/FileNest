# 📂 FileNest - File Metadata CRUD API

This project is a basic **CRUD API in Go** for managing **file metadata** (such as filename, size, type, and upload time).  
It is a foundational backend component for the upcoming **FileNest P2P File Sharing System**.

---

## 📦 Features

- ✅ REST API with Go and Gorilla Mux
- ✅ MongoDB Atlas integration
- ✅ CRUD operations for file metadata
- ✅ Input validation and error handling
- ✅ Manual testing via Postman

---

## 📁 Folder Structure

```
tasks/
└── crud/
├── main.go
├── model/
│ └── models.go
├── controller/
│ └── controller.go
├── router/
│ └── router.go
├── README.md
└── postman_screenshots/
```

## ⚙️ Tech Stack

| Layer          | Tech                          |
|----------------|-------------------------------|
| Language       | Golang 1.21+                  |
| Router         | Gorilla Mux                   |
| Database       | MongoDB Atlas (or local)      |
| Driver         | Mongo Go Driver               |
| Testing Tool   | Postman, Curl                 |

---

## 🚀 Quick Start

### 📋 Prerequisites

- Go 1.21 or higher
- MongoDB Atlas or local MongoDB instance
- Git

### 📥 Installation

```bash
# Download Go dependencies
go mod tidy
```

### ⚙️ Environment Setup

Create a `.env` file in the root with the following:

```
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
```

### ▶️ Run the Server

```bash
go run main.go
```

> Server starts on `http://localhost:4000`

---

## 📡 API Endpoints

| Method | Endpoint            | Description                  |
|--------|---------------------|------------------------------|
| GET    | `/files`        | Get all file records         |
| GET    | `/files/{id}`   | Get file by ID               |
| POST   | `/files`        | Create new file record       |
| PUT    | `/files/{id}`   | Update file by ID            |
| DELETE | `/files/{id}`   | Delete file by ID            |

---

## 📄 Data Model (FileMetadata)

```go
type FileMetadata struct {
  ID          primitive.ObjectID `json:"id,omitempty" bson:"_id,omitempty"`
  Filename    string             `json:"filename" bson:"filename"`
  Size    int64              `json:"filesize" bson:"filesize"`
  ContentType string             `json:"content_type" bson:"content_type"`
  UploadedAt  primitive.DateTime `json:"uploaded_at,omitempty" bson:"uploaded_at,omitempty"
}
```

---

## 🧪 Testing

Use [Postman](https://www.postman.com/) or Curl:

### Create File (POST)

```bash
curl -X POST http://localhost:4000/files \
  -H "Content-Type: application/json" \
  -d '{"filename":"test.pdf","filepath":"/docs/test.pdf","filesize":1024,"content_type":"application/pdf"}'
```

### Get All Files (GET)

```bash
curl http://localhost:4000/files
```

### Update File (PUT)

```bash
curl -X PUT http://localhost:4000/files/<id> \
  -H "Content-Type: application/json" \
  -d '{"filename":"updated.pdf"}'
```

### Delete File (DELETE)

```bash
curl -X DELETE http://localhost:4000/files/<id>
```

