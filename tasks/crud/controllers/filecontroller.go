package controllers

import (
	"context"
	"crud-api/models"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	dbName  = "FileDb"
	colName = "Files"
)

var collection *mongo.Collection

// Initializes MongoDB connection and sets the collection.
func init() {
	if err := godotenv.Load(); err != nil {
		log.Fatal("Error loading .env file")
	}

	connectionString := os.Getenv("MONGODB_URI")
	clientOptions := options.Client().ApplyURI(connectionString)

	client, err := mongo.Connect(context.TODO(), clientOptions)
	if err != nil {
		log.Fatal("Mongo connection error:", err)
	}

	fmt.Println("MongoDB connection successful")
	collection = client.Database(dbName).Collection(colName)
}

// Inserts a new file document into MongoDB.
func createFile(file *models.FileMetadata) error {
	file.ID = primitive.NewObjectID()
	file.CreatedAt = time.Now()
	file.UpdatedAt = time.Now()

	_, err := collection.InsertOne(context.Background(), file)
	return err
}

// Updates a file document by ID with non-empty fields and returns updated doc.
func updateFile(file *models.FileMetadata, id string) error {
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return fmt.Errorf("invalid ID: %v", err)
	}

	updateFields := bson.M{}
	if file.FileName != "" {
		updateFields["filename"] = file.FileName
	}
	if file.FilePath != "" {
		updateFields["filepath"] = file.FilePath
	}
	if file.FileSize != 0 {
		updateFields["filesize"] = file.FileSize
	}
	if file.ContentType != "" {
		updateFields["content_type"] = file.ContentType
	}
	updateFields["updated_at"] = time.Now()

	_, err = collection.UpdateOne(context.Background(), bson.M{"_id": objID}, bson.M{"$set": updateFields})
	if err != nil {
		return err
	}

	return collection.FindOne(context.Background(), bson.M{"_id": objID}).Decode(file)
}

// Deletes a single file document by ID.
func deleteOneFile(fileID string) error {
	objID, err := primitive.ObjectIDFromHex(fileID)
	if err != nil {
		return err
	}

	_, err = collection.DeleteOne(context.Background(), bson.M{"_id": objID})
	return err
}

// Retrieves all file documents from the collection.
func getAllFiles() ([]models.FileMetadata, error) {
	cursor, err := collection.Find(context.Background(), bson.M{})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(context.Background())

	var files []models.FileMetadata
	for cursor.Next(context.Background()) {
		var file models.FileMetadata
		if err := cursor.Decode(&file); err != nil {
			log.Println("Decode error:", err)
			continue
		}
		files = append(files, file)
	}

	return files, nil
}

// Retrieves a single file document by ID.
func getFileByID(fileID string) (*models.FileMetadata, error) {
	objID, err := primitive.ObjectIDFromHex(fileID)
	if err != nil {
		return nil, err
	}

	var file models.FileMetadata
	err = collection.FindOne(context.Background(), bson.M{"_id": objID}).Decode(&file)
	return &file, err
}

// HTTP: Returns all files.
func GetAllFiles(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	files, err := getAllFiles()
	if err != nil {
		http.Error(w, "Failed to fetch files", http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(files)
}

// HTTP: Returns a file by ID.
func GetFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	id := mux.Vars(r)["id"]

	file, err := getFileByID(id)
	if err != nil {
		http.Error(w, "File not found", http.StatusNotFound)
		return
	}

	json.NewEncoder(w).Encode(file)
}

// HTTP: Creates a new file.
func CreateFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	var file models.FileMetadata
	if err := json.NewDecoder(r.Body).Decode(&file); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if err := createFile(&file); err != nil {
		http.Error(w, "Failed to create file", http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(file)
}

// HTTP: Updates a file by ID.
func UpdateFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	id := mux.Vars(r)["id"]

	var file models.FileMetadata
	if err := json.NewDecoder(r.Body).Decode(&file); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if err := updateFile(&file, id); err != nil {
		http.Error(w, "Failed to update file", http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(file)
}

// HTTP: Deletes a file by ID.
func DeleteFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	id := mux.Vars(r)["id"]

	if err := deleteOneFile(id); err != nil {
		http.Error(w, "Failed to delete file", http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(map[string]string{"deleted_id": id})
}
