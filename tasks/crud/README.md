# 📦 FileNest — Go + MongoDB CRUD API

A lightweight, clean RESTful API built using **Golang** and **MongoDB Atlas** for managing file metadata. Designed as a modular, scalable microservice for operations like file uploads, storage indexing, or integrations with larger systems like content managers and intelligent search frameworks.

---

## 🌟 Overview

This service provides simple REST endpoints to perform **Create**, **Read (all/single)**, **Update**, and **Delete** operations on a MongoDB collection storing file metadata.  

Built using native `net/http` for lightweight routing and the official **MongoDB Go Driver**, it’s minimal yet production-ready.

---

## 🎯 Key Features

- 📡 RESTful API with appropriate HTTP methods and status codes  
- 🔐 Clean MongoDB Atlas integration via official Go driver  
- ⚡ Efficient BSON data handling  
- 📝 Modular code organization (handlers, models, config)  
- 📬 API tested via Postman  

---

## 🏗️ Project Structure

filenest/
├── crud/
│ ├── db.go
│ ├── handlers.go
│ ├── models.go
│ ├── main.go
│ ├── go.mod, go.sum
│ └── README.md


---

## ⚙️ Tech Stack

| Layer          | Tech                          |
|:---------------|:-----------------------------|
| Language        | Golang 1.24+                  |
| Database        | MongoDB Atlas                 |
| Driver          | MongoDB Go Driver v1.17.4     |
| Routing         | Native net/http               |
| Testing Tool    | Postman                       |

---

## 🚀 Getting Started

### 📋 Prerequisites

- Go 1.24+
- MongoDB Atlas or local MongoDB instance
- Git

---

### 📥 Installation & Run

```bash
# Clone this repository
git clone <repo-url>
cd <project-folder>

# Install dependencies
go mod tidy

# Set your MongoDB connection URI
export MONGODB_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net/?retryWrites=true&w=majority

# Run the server
go run .

Server runs at: http://localhost:8080

## 📡 API Endpoints

| Method   | Endpoint             | Description                |
|:---------|:---------------------|:---------------------------|
| `GET`    | `/api/files`         | Fetch all files             |
| `GET`    | `/api/files?id=<id>` | Fetch a single file by ID   |
| `POST`   | `/api/files`         | Create new file metadata    |
| `PUT`    | `/api/files?id=<id>` | Update file metadata by ID  |
| `DELETE` | `/api/files?id=<id>` | Delete file by ID           |

---

## 📄 Data Model (FileMetadata)

```go
type FileMetadata struct {
  ID          string    `json:"id" bson:"_id,omitempty"`
  FileName    string    `json:"filename" bson:"filename"`
  FilePath    string    `json:"filepath" bson:"filepath"`
  FileSize    int64     `json:"filesize" bson:"filesize"`
  ContentType string    `json:"content_type" bson:"content_type"`
  CreatedAt   time.Time `json:"created_at" bson:"created_at"`
  UpdatedAt   time.Time `json:"updated_at" bson:"updated_at"`
}



🧪 Testing (Postman)
📥 Create File (POST)

POST http://localhost:8080/api/files
{
  "filename": "example.png",
  "filepath": "/images/example.png",
  "filesize": 2048,
  "content_type": "image/png"
}

📄 Get All Files (GET)

GET http://localhost:8080/api/files
📄 Get Single File (GET)

GET http://localhost:8080/api/files?id=<id>
✏️ Update File (PUT)

PUT http://localhost:8080/api/files?id=<id>
{
  "filename": "updated.png",
  "filepath": "/images/updated.png",
  "filesize": 4096,
  "content_type": "image/jpeg"
}

❌ Delete File (DELETE)

DELETE http://localhost:8080/api/files?id=<id>
