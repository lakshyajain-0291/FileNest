package main

import (
	"context"
	"encoding/json" //for encoding/decoding json requests and responses
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

func createFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var file FileMetadata
	err := json.NewDecoder(r.Body).Decode(&file)
	if err != nil {
		fmt.Println("Error decoding request body:", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	file.CreatedAt = time.Now()
	file.UpdatedAt = time.Now()
	insertResult, err := fileCollection.InsertOne(context.TODO(), file)
	if err != nil {
		fmt.Println("Error inserting into database:", err)
		http.Error(w, "Failed to insert data", http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(insertResult.InsertedID)
}

func getAllFiles(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	cursor, err := fileCollection.Find(context.TODO(), bson.D{})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer cursor.Close(context.TODO()) //ensure the cursor is closed after use

	var files []FileMetadata

	for cursor.Next(context.TODO()) {
		var file FileMetadata
		_ = cursor.Decode(&file)
		files = append(files, file)
	}
	json.NewEncoder(w).Encode(files)
}

func getFileByID(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	params := mux.Vars(r)
	id, _ := primitive.ObjectIDFromHex(params["id"])

	var file FileMetadata
	err := fileCollection.FindOne(context.TODO(), bson.M{"_id": id}).Decode(&file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	json.NewEncoder(w).Encode(file)
}

func updateFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	params := mux.Vars(r)
	id, _ := primitive.ObjectIDFromHex(params["id"])

	var updated FileMetadata
	_ = json.NewDecoder(r.Body).Decode(&updated)
	updated.UpdatedAt = time.Now()

	update := bson.M{"$set": updated}
	result, err := fileCollection.UpdateOne(context.TODO(), bson.M{"_id": id}, update)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(result)
}

func deleteFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	params := mux.Vars(r)
	id, _ := primitive.ObjectIDFromHex(params["id"])

	result, err := fileCollection.DeleteOne(context.TODO(), bson.M{"_id": id})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(result)
}
