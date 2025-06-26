package router

import (
	"crud-api/controllers"

	"github.com/gorilla/mux"
)

func Router() *mux.Router {
	router := mux.NewRouter()

	// File API routes
	router.HandleFunc("/api/files", controllers.GetAllFiles).Methods("GET")        // Get all files
	router.HandleFunc("/api/files/{id}", controllers.GetFile).Methods("GET")       // Get single file by ID
	router.HandleFunc("/api/files", controllers.CreateFile).Methods("POST")        // Create a new file
	router.HandleFunc("/api/files/{id}", controllers.UpdateFile).Methods("PUT")    // Update a file by ID
	router.HandleFunc("/api/files/{id}", controllers.DeleteFile).Methods("DELETE") // Delete a file by ID\

	return router
}
