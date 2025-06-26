package router

import (
	"file-nest-crud/controller"
	"github.com/gorilla/mux"
)

func Router() *mux.Router {
	router := mux.NewRouter()

	router.HandleFunc("/api/files", controller.GetAllRecords).Methods("GET")
	router.HandleFunc("/api/files/{id}", controller.GetSingleRecord).Methods("GET")
	router.HandleFunc("/api/files", controller.CreateRecord).Methods("POST")
	router.HandleFunc("/api/files/{id}", controller.UpdateRecord).Methods("PUT")
	router.HandleFunc("/api/files/{id}", controller.DeleteRecord).Methods("DELETE")

	return router
}
