package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"file-nest-crud/model"
	"github.com/gorilla/mux"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

const connectionString = "mongodb+srv://ishatvarshney:Kg438PRRVxc6vg5@cluster0.rygxhix.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
const dbName = "FileNest"
const colName = "Task1"

var collection *mongo.Collection

func init() {
	client, err := mongo.Connect(options.Client().ApplyURI(connectionString))
	if err != nil {
		log.Fatal(err)
	}

	err = client.Ping(context.Background(), nil)
	if err != nil {
		log.Fatal("MongoDB ping failed:", err)
	}

	fmt.Println("MongoDB connected successfully")
	collection = client.Database(dbName).Collection(colName)
	fmt.Println("Collection instance is ready")
}

func validateRecord(record model.FileMetadata) error {
	if strings.TrimSpace(record.FileName) == "" {
		return errors.New("file name is required")
	}
	if strings.TrimSpace(record.FilePath) == "" {
		return errors.New("file path is required")
	}
	if record.FileSize <= 0 {
		return errors.New("file size must be greater than zero")
	}
	if strings.TrimSpace(record.ContentType) == "" {
		return errors.New("content type is required")
	}
	return nil
}

func createRecord(record model.FileMetadata) {
	inserted, err := collection.InsertOne(context.Background(), record)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Inserted a record with id:", inserted.InsertedID)
}

func updateRecord(ID int, updatedData model.FileMetadata) {
	filter := bson.M{"id": ID}
	update := bson.M{
		"$set": bson.M{
			"filepath":     updatedData.FilePath,
			"filesize":     updatedData.FileSize,
			"content_type": updatedData.ContentType,
			"updated_at":   time.Now(),
		},
	}
	result, err := collection.UpdateOne(context.Background(), filter, update)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Modified count:", result.ModifiedCount)
}

func deleteRecord(ID int) {
	filter := bson.M{"id": ID}
	result, err := collection.DeleteOne(context.Background(), filter)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Record was deleted with count:", result.DeletedCount)
}

func getallRecords() []bson.M {
	cur, err := collection.Find(context.Background(), bson.D{{}})
	if err != nil {
		log.Fatal(err)
	}

	var records []bson.M
	for cur.Next(context.Background()) {
		var record bson.M
		err := cur.Decode(&record)
		if err != nil {
			log.Fatal(err)
		}
		records = append(records, record)
	}
	defer cur.Close(context.Background())
	return records
}

func getsingleRecord(ID int) bson.M {
	filter := bson.M{"id": ID}
	var record bson.M
	err := collection.FindOne(context.Background(), filter).Decode(&record)
	if err != nil {
		log.Println("Error finding record:", err)
		return bson.M{"error": "Record not found"}
	}
	return record
}

func GetAllRecords(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	allrecords := getallRecords()
	json.NewEncoder(w).Encode(allrecords)
}

func GetSingleRecord(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	params := mux.Vars(r)
	ID, err := strconv.Atoi(params["id"])
	if err != nil {
		http.Error(w, "Invalid ID", http.StatusBadRequest)
		return
	}
	singlerecord := getsingleRecord(ID)
	json.NewEncoder(w).Encode(singlerecord)
}

func CreateRecord(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Methods", "POST")

	var record model.FileMetadata
	err := json.NewDecoder(r.Body).Decode(&record)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	err = validateRecord(record)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	record.CreatedAt = time.Now()
	record.UpdatedAt = time.Now()

	createRecord(record)
	json.NewEncoder(w).Encode(record)
}

func UpdateRecord(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Methods", "PUT")

	params := mux.Vars(r)
	id, err := strconv.Atoi(params["id"])
	if err != nil {
		http.Error(w, "Invalid ID", http.StatusBadRequest)
		return
	}

	var record model.FileMetadata
	err = json.NewDecoder(r.Body).Decode(&record)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	err = validateRecord(record)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	record.UpdatedAt = time.Now()
	updateRecord(id, record)
	json.NewEncoder(w).Encode(record)
}

func DeleteRecord(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Methods", "DELETE")

	params := mux.Vars(r)
	id, err := strconv.Atoi(params["id"])
	if err != nil {
		http.Error(w, "Invalid ID", http.StatusBadRequest)
		return
	}
	deleteRecord(id)
	json.NewEncoder(w).Encode(bson.M{"deleted": id})
}
