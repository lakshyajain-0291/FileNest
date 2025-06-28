package router

import (
	"mongodbgo/controller"

	"github.com/gorilla/mux"
)

func Router() *mux.Router {
	router := mux.NewRouter()

	router.HandleFunc("/files", controller.CreateFile).Methods("POST")
	router.HandleFunc("/files", controller.GetAllFiles).Methods("GET")
	router.HandleFunc("/files/{id}", controller.GetFile).Methods("GET")
	router.HandleFunc("/files/{id}", controller.UpdateFile).Methods("PUT")
	router.HandleFunc("/files/{id}", controller.DeleteFile).Methods("DELETE")

	return router
}
