package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"mongodbgo/model"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const connectionString = "mongodb+srv://Anarghya:hellomonkey@cluster0.aprqfub.mongodb.net/"
const dbName = "filenest"
const colName = "metadata"

var collection *mongo.Collection

func init() {
	clientOption := options.Client().ApplyURI(connectionString)
	client, err := mongo.Connect(context.TODO(), clientOption)

	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(("MongoDB Connection Successful"))

	collection = client.Database(dbName).Collection(colName)
	fmt.Println("Collection instance is ready")
}

func CreateFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var file model.FileMeta
	err := json.NewDecoder(r.Body).Decode(&file)
	if err != nil || file.FileName == "" || file.Size <= 0 {
		http.Error(w, "Invalid input", http.StatusBadRequest)
		return
	}

	file.UploadedAt = primitive.NewDateTimeFromTime(time.Now())
	result, err := collection.InsertOne(context.TODO(), file)
	if err != nil {
		http.Error(w, "Insert failed", http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(result)
}

func GetAllFiles(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	cur, err := collection.Find(context.TODO(), bson.D{{}})
	if err != nil {
		http.Error(w, "Error fetching", http.StatusInternalServerError)
		return
	}
	var files []bson.M
	if err := cur.All(context.TODO(), &files); err != nil {
		http.Error(w, "Decode error", http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(files)
}

func GetFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	id, _ := primitive.ObjectIDFromHex(mux.Vars(r)["id"])
	var file bson.M
	err := collection.FindOne(context.TODO(), bson.M{"_id": id}).Decode(&file)
	if err != nil {
		http.Error(w, "Not found", http.StatusNotFound)
		return
	}
	json.NewEncoder(w).Encode(file)
}

func UpdateFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	id, _ := primitive.ObjectIDFromHex(mux.Vars(r)["id"])

	var updated model.FileMeta
	if err := json.NewDecoder(r.Body).Decode(&updated); err != nil {
		http.Error(w, "Invalid input", http.StatusBadRequest)
		return
	}

	update := bson.M{"$set": bson.M{
		"filename":     updated.FileName,
		"size":         updated.Size,
		"content_type": updated.ContentType,
	}}

	result, err := collection.UpdateOne(context.TODO(), bson.M{"_id": id}, update)
	if err != nil {
		http.Error(w, "Update failed", http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(result)
}

func DeleteFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	id, _ := primitive.ObjectIDFromHex(mux.Vars(r)["id"])

	result, err := collection.DeleteOne(context.TODO(), bson.M{"_id": id})
	if err != nil {
		http.Error(w, "Delete failed", http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(result)
}
