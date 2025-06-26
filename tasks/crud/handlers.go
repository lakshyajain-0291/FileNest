package main

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

// Create a file
func createFile(w http.ResponseWriter, r *http.Request) {
	var file FileMetadata
	_ = json.NewDecoder(r.Body).Decode(&file)
	file.CreatedAt = time.Now()
	file.UpdatedAt = time.Now()

	res, err := collection.InsertOne(context.TODO(), file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(res)
}

// Get all files
func getAllFiles(w http.ResponseWriter, r *http.Request) {
	cursor, err := collection.Find(context.TODO(), bson.M{})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer cursor.Close(context.TODO())

	var files []FileMetadata
	for cursor.Next(context.TODO()) {
		var file FileMetadata
		cursor.Decode(&file)
		files = append(files, file)
	}
	json.NewEncoder(w).Encode(files)
}

// Get file by ID
func getFile(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	objID, _ := primitive.ObjectIDFromHex(id)

	var file FileMetadata
	err := collection.FindOne(context.TODO(), bson.M{"_id": objID}).Decode(&file)
	if err != nil {
		http.Error(w, "File not found", http.StatusNotFound)
		return
	}
	json.NewEncoder(w).Encode(file)
}

// Update a file
func updateFile(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	objID, _ := primitive.ObjectIDFromHex(id)

	var file FileMetadata
	_ = json.NewDecoder(r.Body).Decode(&file)
	file.UpdatedAt = time.Now()

	update := bson.M{
		"$set": file,
	}

	_, err := collection.UpdateOne(context.TODO(), bson.M{"_id": objID}, update)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Write([]byte("File updated successfully"))
}

// Delete a file
func deleteFile(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	objID, _ := primitive.ObjectIDFromHex(id)

	_, err := collection.DeleteOne(context.TODO(), bson.M{"_id": objID})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Write([]byte("File deleted successfully"))
}
