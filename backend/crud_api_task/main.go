package main //sort of an enry point which helps in turning the entire file into an executable binary

import (
	"log"      //for logging errors/server messages
	"net/http" //built in server library for Go, to handle requests and start server

	"github.com/gorilla/mux" //popular routing go library, useful for syntax like /files/{id}
)

func main() {
	router := mux.NewRouter() //new router instance to control all requests

	route := router.PathPrefix("/api").Subrouter() //tells us that this router is handling all api requests, instead of justs starting with /files

	//All Necessary REST API endpoints

	route.HandleFunc("/files", createFile).Methods("POST")
	route.HandleFunc("/files", getAllFiles).Methods("GET")
	route.HandleFunc("/files/{id}", getFileByID).Methods("GET")
	route.HandleFunc("/files/{id}", updateFile).Methods("PUT")
	route.HandleFunc("/files/{id}", deleteFile).Methods("DELETE")

	log.Fatal(http.ListenAndServe(":8000", router)) //strts server at port 8000 and prints any errors that doesnt allow serevr to start
}
