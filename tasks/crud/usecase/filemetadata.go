package usecase

//controls what happens on the local server when a request is made to the file service
import (
	"encoding/json"
	"log"
	"math/rand"
	"net/http"
	"strconv"

	"github.com/centauri1219/filenest/tasks/crud/model"
	"github.com/centauri1219/filenest/tasks/crud/repository"
	"github.com/gorilla/mux"
	"go.mongodb.org/mongo-driver/mongo"
)

type FileService struct {
	MongoCollection *mongo.Collection
}

type Response struct {
	Data  interface{} `json:"data,omitempty"`
	Error string      `json:"error,omitempty"`
}

func (svc *FileService) CreateFile(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Content-Type", "application/json") //indicates JSON output
	res := &Response{}                                 //to hold result or error
	defer json.NewEncoder(w).Encode(res)

	var fil model.FileMetadata

	err := json.NewDecoder(r.Body).Decode(&fil)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		log.Println("invalid body", err)
		res.Error = err.Error()
		return
	}

	//assign new fileid
	fil.ID = rand.Int() //for simplicity, using a random number as file ID

	repo := repository.FileMetadataRepo{
		MongoCollection: svc.MongoCollection}

	//inser file
	insertID, err := repo.InsertFile(&fil) //calls repository function to insert file metadata into the database
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		log.Println("error while inserting file", err)
		res.Error = err.Error()
		return
	}

	res.Data = fil.ID
	w.WriteHeader(http.StatusOK)
	log.Println("file created successfully with ID:", insertID, fil)
}

func (svc *FileService) GetFilebyID(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Content-Type", "application/json")
	res := &Response{}
	defer json.NewEncoder(w).Encode(res)
	//get file id
	fileIDStr := mux.Vars(r)["id"]         //extracts the file ID from the URL path
	fileID, err := strconv.Atoi(fileIDStr) //convert to integer
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		log.Println("invalid file id", err)
		res.Error = "invalid file id"
		return
	}
	log.Println("file id:", fileID)
	repo := repository.FileMetadataRepo{
		MongoCollection: svc.MongoCollection}

	fil, err := repo.FindFilebyID(fileID)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		log.Println("error while getting file by ID", err)
		res.Error = err.Error()
		return
	}

	res.Data = fil
	w.WriteHeader(http.StatusOK)

}
func (svc *FileService) GetAllFiles(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Content-Type", "application/json")
	res := &Response{}
	defer json.NewEncoder(w).Encode(res)
	repo := repository.FileMetadataRepo{
		MongoCollection: svc.MongoCollection}

	fil, err := repo.FindAllFiles()
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		log.Println("error while getting file by ID", err)
		res.Error = err.Error()
		return
	}

	res.Data = fil
	w.WriteHeader(http.StatusOK)
}
func (svc *FileService) UpdateFilebyID(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Content-Type", "application/json")
	res := &Response{}
	defer json.NewEncoder(w).Encode(res)

	//get file id
	fileIDStr := mux.Vars(r)["id"]
	fileID, err := strconv.Atoi(fileIDStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		log.Println("invalid file id", err)
		res.Error = "invalid file id"
		return
	}
	log.Println("file id:", fileID)
	if fileIDStr == "" {
		w.WriteHeader(http.StatusBadRequest)
		log.Println("file ID is required")
		res.Error = "file ID is required"
		return
	}

	var fil model.FileMetadata
	err = json.NewDecoder(r.Body).Decode(&fil)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		log.Println("invalid body", err)
		res.Error = err.Error()
		return
	}

	fil.ID = fileID //assign the file ID from the URL
	repo := repository.FileMetadataRepo{MongoCollection: svc.MongoCollection}
	count, err := repo.UpdateFilebyID(fileID, &fil)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		log.Println("error while updating file", err)
		res.Error = err.Error()
		return
	}
	res.Data = count
	w.WriteHeader(http.StatusOK)
}

func (svc *FileService) DeleteFilebyID(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Content-Type", "application/json")
	res := &Response{}
	defer json.NewEncoder(w).Encode(res)

	//get file id
	fileIDStr := mux.Vars(r)["id"]
	fileID, err := strconv.Atoi(fileIDStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		log.Println("invalid file id", err)
		res.Error = "invalid file id"
		return
	}
	log.Println("file id:", fileID)

	repo := repository.FileMetadataRepo{
		MongoCollection: svc.MongoCollection}
	count, err := repo.DeleteFilebyID(fileID)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		log.Println("error while deleting file", err)
		res.Error = err.Error()
		return
	}

	res.Data = count
	w.WriteHeader(http.StatusOK)
}
func (svc *FileService) DeleteAll(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Content-Type", "application/json")
	res := &Response{}
	defer json.NewEncoder(w).Encode(res)

	//get file id

	repo := repository.FileMetadataRepo{
		MongoCollection: svc.MongoCollection}
	count, err := repo.DeleteAll()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		log.Println("error while deleting file", err)
		res.Error = err.Error()
		return
	}

	res.Data = count
	w.WriteHeader(http.StatusOK)
}
