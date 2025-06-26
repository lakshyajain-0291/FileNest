package main

import (
	"crud-api/pkg/routes"
	"fmt"
	"log"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)
func main(){
	// Load the dotenv.
	err := godotenv.Load()
    if err != nil {
        log.Println("Warning: .env file not found or could not be loaded")
    }

	router := gin.Default()
	routes.RegisterFileRoutes(router)
	
	port := os.Getenv("PORT")
	addr := fmt.Sprintf(":%v", port)
	router.Run(addr) 
}