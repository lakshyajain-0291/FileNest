package repository

import (
	"context"
	"log"
	"math/rand"
	"testing"

	"github.com/centauri1219/filenest/tasks/crud/model"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
)

func newMongoClient() *mongo.Client {
	mongoTestClient, err := mongo.Connect(context.Background(),
		options.Client().ApplyURI("mongodb+srv://admin:49c14IwyVUTJsqkE@cluster0.4k6j8iy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"))

	if err != nil {
		log.Fatal("error while connecting mongodb", err)
	}

	log.Println("Mongodb succesfully connected")

	err = mongoTestClient.Ping(context.Background(), readpref.Primary())

	if err != nil {
		log.Fatal("ping failed", err)
	}

	log.Println("ping success")

	return mongoTestClient
}

func TestMongoOpearations(t *testing.T) {
	mongoTestClient := newMongoClient()
	defer mongoTestClient.Disconnect(context.Background())

	//dummy data
	file1 := rand.Int()
	file2 := rand.Int()

	// we have just created client we have to connect to our collection
	coll := mongoTestClient.Database("fileinfodb").Collection("files_test")

	filerepo := FileMetadataRepo{MongoCollection: coll}

	//Insert 1 file data
	t.Run("Insert File 1", func(t *testing.T) {
		fil := model.FileMetadata{
			ID:          file1,
			FileName:    "Testfile1",
			FilePath:    "C:/men",
			FileSize:    6969,
			ContentType: ".txt",
			//leaving time shit empty
		}
		result, err := filerepo.InsertFile(&fil)
		if err != nil {
			t.Fatal("insert 1 opertaion failed", err)
		}
		t.Log("Insert 1 success", result)
	})

	t.Run("Insert File 2", func(t *testing.T) {
		fil := model.FileMetadata{
			ID:          file2,
			FileName:    "Testfile2",
			FilePath:    "C:/men2",
			FileSize:    6969,
			ContentType: ".txt",
			//leaving time shit empty
		}
		result, err := filerepo.InsertFile(&fil)
		if err != nil {
			t.Fatal("insert 2 opertaion failed", err)
		}
		t.Log("Insert 2 success", result)
	})

	//get file 1 data
	t.Run("Get File 1", func(t *testing.T) {
		result, err := filerepo.FindFilebyID(file1)

		if err != nil {
			t.Fatal("get operation failed", err)
		}
		t.Log("Get File 1 success", result.FileName)
	})

	t.Run("Get all files", func(t *testing.T) {
		results, err := filerepo.FindAllFiles()

		if err != nil {
			t.Fatal("get all operation failed", err)
		}
		t.Log("Get all files success", results)
	})

	t.Run("Update File1 name", func(t *testing.T) {
		fil := model.FileMetadata{
			ID:          file1,
			FileName:    "UpdatedTestfile1",
			FilePath:    "C:/men",
			FileSize:    6969,
			ContentType: ".txt",
		}
		result, err := filerepo.UpdateFilebyID(file1, &fil)
		if err != nil {
			t.Fatal("update operation failed", err)
		}
		t.Log("Update File1 name success", result)
	})

	t.Run("Get file 1 after update", func(t *testing.T) {
		result, err := filerepo.FindFilebyID(file1)

		if err != nil {
			t.Fatal("get operation after update failed", err)
		}
		t.Log("Get File 1 after update success", result.FileName)
	})

	//delete file 1
	t.Run("Delete File 1", func(t *testing.T) {
		result, err := filerepo.DeleteFilebyID(file1)
		if err != nil {
			log.Fatal("delete operation failed", err)
		}
		t.Log("Delete File 1 success", result)
	})

	t.Run("Get all files after delete", func(t *testing.T) {
		results, err := filerepo.FindAllFiles()
		if err != nil {
			log.Fatal("get all operation failed", err)
		}
		t.Log("Get all files after delete success", results)
	})

	//delete all files
	t.Run("Delete all files for cleanup", func(t *testing.T) {
		result, err := filerepo.DeleteAll()
		if err != nil {
			log.Fatal("delete all operation failed", err)
		}

		t.Log("Delete all files success", result)
	})
}
