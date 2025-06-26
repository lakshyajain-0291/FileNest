package database

import (
	"filenest-xs/model"
	"fmt"
	"os"

	"github.com/joho/godotenv"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// DbInit establishes a connection to the PostgreSQL database,
func DbInit() (*gorm.DB, error) {
	err := godotenv.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load .env file: %w", err)
	}
	dsn := fmt.Sprintf("postgres://%v:%v@%v:%v/%v?sslmode=disable",
		os.Getenv("POSTGRES_USER"),
		os.Getenv("POSTGRES_PASS"),
		os.Getenv("POSTGRES_HOST"),
		os.Getenv("POSTGRES_PORT"),
		os.Getenv("POSTGRES_DB_NAME"),
	)

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("failed to get generic DB object: %w", err)
	}

	if err := sqlDB.Ping(); err != nil {
		return nil, fmt.Errorf("database ping failed: %w", err)
	}

	fmt.Println("Database connection successful!")
	return db, nil
}

// UpsertFileIndex inserts or updates a FileIndex record in the database.
func UpsertFileIndex(db *gorm.DB, fileIndex *model.FileIndex) error {
	result := db.Clauses(
		clause.OnConflict{
			Columns: []clause.Column{
				{Name: "file_name"},
				{Name: "file_path"},
			},
			DoUpdates: clause.Assignments(map[string]interface{}{
				"embedding":  fileIndex.Embedding,
				"d1tv_id":    fileIndex.D1TVID,
				"indexed_at": fileIndex.IndexedAt,
			}),
		},
	).Create(fileIndex)

	if result.Error != nil {
		return result.Error
	}

	return nil
}
