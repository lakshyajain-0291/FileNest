package main

import (
	"fmt"
	"net/http"
)

func main() {
	ConnectDB()

	http.HandleFunc("/api/files", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "GET":
			id := r.URL.Query().Get("id")
			if id == "" {
				getAllFiles(w, r)
			} else {
				getFile(w, r)
			}
		case "POST":
			createFile(w, r)
		case "PUT":
			updateFile(w, r)
		case "DELETE":
			deleteFile(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	fmt.Println("Server started at :8080")
	http.ListenAndServe(":8080", nil)
}
