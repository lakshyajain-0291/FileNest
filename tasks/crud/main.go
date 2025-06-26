package main

import (
	"crud-api/router"
	"fmt"
	"net/http"
	"os"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	fmt.Println("Server running at http://localhost:" + port)
	err := http.ListenAndServe(":"+port, router.Router())
	if err != nil {
		fmt.Println("Failed to start server:", err)
	}
}
