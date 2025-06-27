package db

import (
	"log"
	"os" //to read environment variables

	"backend_prototype/models"

	"gorm.io/driver/postgres"
	"gorm.io/gorm" //main GORM library
)

var DB *gorm.DB //pointer to GORM database connection

func InitDB() {
	dsn := os.Getenv("POSTGRES_DSN") //dsn (Data Source Name) connection string to PostgreSQL, set as environment variable
	var err error
	DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{}) //here, postgres.open creates a gorm drive for postgres and gorm.open establishes the actual connection
	if err != nil {
		log.Fatalf("Failed to connect DB: %v", err)
	}
	DB.AutoMigrate(&models.FileIndex{}) //telling gorm to update the database schema to match the FileIndex model
}
