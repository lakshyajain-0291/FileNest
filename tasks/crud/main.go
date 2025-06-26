package main

import (
	"context"
	"log"
	"net/http"
	"os"

	"github.com/centauri1219/filenest/tasks/crud/usecase"
	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
)

var mongoClient *mongo.Client

func init() { //initiate the mongo conneciton
	//load .env file
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file", err)
	}
	log.Println("Environment variables loaded successfully ")

	//create mongo client
	mongoClient, err = mongo.Connect(context.Background(), options.Client().ApplyURI(os.Getenv("MONGO_URI"))) //mongo connect only tells if the connection string is correct
	//context.background() is used to create a context for the connection, which can be used to cancel the operation if needed and manage its lifetime
	if err != nil {
		log.Fatal("Error connecting to MongoDB", err)
	}

	err = mongoClient.Ping(context.Background(), readpref.Primary()) //sends request to mongo client to verify if it can connect to deployment
	if err != nil {
		log.Fatal("Error pinging MongoDB", err)
	}

	log.Println("MongoDB connected successfully")
}

func main() {
	defer mongoClient.Disconnect(context.Background())                                          //close mongo connection
	coll := mongoClient.Database(os.Getenv("DB_NAME")).Collection(os.Getenv("COLLECTION_NAME")) //coll is a reference to the mongodb collection using mongoclient
	//create file service
	fileservice := usecase.FileService{MongoCollection: coll} //fileservice is an object that provides file management logic, backed by mongodb

	//create router
	r := mux.NewRouter()                                           //directs incoming http requests to the handler funcs
	r.HandleFunc("/health", healthHandler).Methods(http.MethodGet) //check if server is running
	r.HandleFunc("/file", fileservice.CreateFile).Methods(http.MethodPost)
	r.HandleFunc("/file/{id}", fileservice.GetFilebyID).Methods(http.MethodGet)
	r.HandleFunc("/file", fileservice.GetAllFiles).Methods(http.MethodGet)
	r.HandleFunc("/file/{id}", fileservice.UpdateFilebyID).Methods(http.MethodPut)
	r.HandleFunc("/file/{id}", fileservice.DeleteFilebyID).Methods(http.MethodDelete)
	r.HandleFunc("/file", fileservice.DeleteAll).Methods(http.MethodDelete)

	log.Println("Server is running on port 4444")
	http.ListenAndServe(":4444", r)
}

func healthHandler(w http.ResponseWriter, r *http.Request) { //mapped to /health endpoint
	w.WriteHeader(http.StatusOK)   //http response
	w.Write([]byte("running....")) //write response to client
}
