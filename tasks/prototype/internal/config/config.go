package config

//handles configuration settings for the FileNest prototype, loading them from .env file
import (
	"log"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type Config struct {
	DatabaseURL    string
	WorkerCount    int
	ProcessTimeout int // seconds
	EmbeddingDim   int
}

func LoadConfig() *Config {
	if err := godotenv.Load(); err != nil { // loads environment variables from a .env file if it exists
		log.Println("No .env file found, using environment variables")
	}

	return &Config{ // pointer to a Config struct
		DatabaseURL:    getEnv("DATABASE_URL", "postgres://user:password@localhost:5432/filenest?sslmode=disable"), //fetches the env variable or used a default value
		WorkerCount:    getEnvAsInt("WORKER_COUNT", 5),
		ProcessTimeout: getEnvAsInt("PROCESS_TIMEOUT", 5),
		EmbeddingDim:   getEnvAsInt("EMBEDDING_DIM", 128),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" { //checks if the environment variable is set
		return value
	}
	return defaultValue //else returns the default value
}

func getEnvAsInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intVal, err := strconv.Atoi(value); err == nil {
			return intVal
		}
	}
	return defaultValue
}
