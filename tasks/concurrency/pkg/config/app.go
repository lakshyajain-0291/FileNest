package config

import (
	"conc-task/pkg/models"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

func InitDB() *gorm.DB {
	err := godotenv.Load()
    if err != nil {
        log.Println("Warning: .env file not found or could not be loaded")
    }
	// user- postgres, pwd- password, host- localhost, port- 5432, database- filenest
	dsn := fmt.Sprintf("postgres://%v:%v@localhost:%v/%v?sslmode=disable",
	 	os.Getenv("DB_USER"),
		os.Getenv("DB_PASSWORD"),
		os.Getenv("POSTGRES_PORT"),
		os.Getenv("DB_NAME"))
	// dsn := "postgres://postgres:password@localhost:5432/filenest?sslmode=disable"
	db,err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if(err != nil){
		panic(err.Error())
	}
	return db
}

func UpsertFile(db *gorm.DB, file models.FileIndex) error {
    return db.Clauses(clause.OnConflict{
        Columns:   []clause.Column{{Name: "id"}}, // or use a unique column/key
        UpdateAll: true,
    }).Create(&file).Error
}