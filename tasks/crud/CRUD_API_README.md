
# 📦 CRUD API with Golang, MongoDB & Gorilla Mux

A lightweight, performant RESTful API in Go designed to handle basic Create, Read, Update, and Delete (CRUD) operations over file metadata using MongoDB as the database and Gorilla Mux as the router.

---

## 🌟 Overview

This project implements a modular and extensible file metadata management service. It provides clean REST endpoints to perform CRUD operations on a MongoDB collection. Designed to be simple yet scalable, the API can be easily integrated into larger systems such as content managers, file explorers, or intelligent search frameworks like FileNest.

---

## 🎯 Key Features

- ⚙️ RESTful API with proper HTTP methods and status codes
- 🛡️ Input validation and error handling
- 🗃️ MongoDB integration with BSON support for efficient storage
- 🔄 Endpoints for Create, Read (all/single), Update, and Delete operations
- 📎 Built with modular code for easy extension
- 📫 Tested with Postman and Curl

---

## 🏗️ Project Architecture

```
crud-api/
├── controllers/       # API logic and handler functions
├── models/            # Data model (MongoDB schema)
├── router/            # Gorilla Mux routing definitions
├── .env               # MongoDB credentials and configs
├── go.mod             # Module dependencies
├── go.sum             # Module checksums
├── main.go            # Application entrypoint
└── README.md          # This file
```

---

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

> Server starts on `http://localhost:8000`

---

## 📡 API Endpoints

| Method | Endpoint            | Description                  |
|--------|---------------------|------------------------------|
| GET    | `/api/files`        | Get all file records         |
| GET    | `/api/files/{id}`   | Get file by ID               |
| POST   | `/api/files`        | Create new file record       |
| PUT    | `/api/files/{id}`   | Update file by ID            |
| DELETE | `/api/files/{id}`   | Delete file by ID            |

---

## 📄 Data Model (FileMetadata)

```go
type FileMetadata struct {
  ID          primitive.ObjectID `json:"id,omitempty" bson:"_id,omitempty"`
  FileName    string             `json:"filename" bson:"filename"`
  FilePath    string             `json:"filepath" bson:"filepath"`
  FileSize    int64              `json:"filesize" bson:"filesize"`
  ContentType string             `json:"content_type" bson:"content_type"`
  CreatedAt   time.Time          `json:"created_at" bson:"created_at"`
}
```

---

## 🧪 Testing

Use [Postman](https://www.postman.com/) or Curl:

### Create File (POST)

```bash
curl -X POST http://localhost:8000/api/files \
  -H "Content-Type: application/json" \
  -d '{"filename":"test.pdf","filepath":"/docs/test.pdf","filesize":1024,"content_type":"application/pdf"}'
```

### Get All Files (GET)

```bash
curl http://localhost:8000/api/files
```

### Update File (PUT)

```bash
curl -X PUT http://localhost:8000/api/files/<id> \
  -H "Content-Type: application/json" \
  -d '{"filename":"updated.pdf"}'
```

### Delete File (DELETE)

```bash
curl -X DELETE http://localhost:8000/api/files/<id>
```

