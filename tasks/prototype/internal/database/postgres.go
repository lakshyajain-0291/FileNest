package database

// manages the pooled connection to a PostgreSQL database, initializes the schema, and provides methods to insert and retrieve file index data
import (
	"context"
	"fmt"
	"time"

	"github.com/centauri1219/FileNest/tasks/prototype/internal/models"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Database struct {
	pool *pgxpool.Pool // connection pool to the PostgreSQL database, pool manages concurrent access
}

func NewDatabase(ctx context.Context, databaseURL string) (*Database, error) {
	config, err := pgxpool.ParseConfig(databaseURL) // parses the dataset URL
	if err != nil {
		return nil, fmt.Errorf("failed to parse database URL: %w", err)
	}

	// Configure connection pool
	config.MaxConns = 10
	config.MinConns = 2
	config.MaxConnLifetime = time.Hour //sets how long connection lives
	config.MaxConnIdleTime = time.Minute * 30

	pool, err := pgxpool.NewWithConfig(ctx, config) //initialises conneciton pool
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	// Test connection
	if err := pool.Ping(ctx); err != nil { //pings the database to ensure it's reachable
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	db := &Database{pool: pool} //new database instance iwthin the pool

	// Initialize schema
	if err := db.initSchema(ctx); err != nil { //to create tavles and indexes if they don't exist
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return db, nil
}

func (db *Database) Close() {
	db.pool.Close() //close all connections in the pool
}

func (db *Database) initSchema(ctx context.Context) error { //create file_index table
	schema := `
    CREATE TABLE IF NOT EXISTS file_index (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL UNIQUE,
        embedding FLOAT8[] NOT NULL,
        d1tv_id INT NOT NULL,
        indexed_at TIMESTAMP DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_file_index_d1tv_id ON file_index(d1tv_id);
    CREATE INDEX IF NOT EXISTS idx_file_index_filepath ON file_index(filepath);
    `

	_, err := db.pool.Exec(ctx, schema) //aquires a connection from the pool
	return err
}

func (db *Database) InsertFileIndex(ctx context.Context, fileIndex *models.FileIndex) error { // inserts a new file index or updates an existing one based on the filepath
	query := `
    INSERT INTO file_index (filename, filepath, embedding, d1tv_id)
    VALUES ($1, $2, $3, $4)
    ON CONFLICT (filepath) DO UPDATE SET
        filename = EXCLUDED.filename,
        embedding = EXCLUDED.embedding,
        d1tv_id = EXCLUDED.d1tv_id,
        indexed_at = now()
    RETURNING id, indexed_at
    `

	err := db.pool.QueryRow( // executes the query and returns the ID and indexed_at timestamp
		ctx, // context for cancellation
		query,
		fileIndex.Filename,
		fileIndex.Filepath,
		fileIndex.Embedding,
		fileIndex.D1TVID,
	).Scan(&fileIndex.ID, &fileIndex.IndexedAt)

	return err
}

func (db *Database) GetFileIndexStats(ctx context.Context) (map[int]int, error) {
	query := `SELECT d1tv_id, COUNT(*) FROM file_index GROUP BY d1tv_id ORDER BY d1tv_id`
	// retrieves the count of indexed files grouped by d1tv_id
	rows, err := db.pool.Query(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	stats := make(map[int]int)
	for rows.Next() {
		var d1tvID, count int // key is d1tv_id and value is count of files
		if err := rows.Scan(&d1tvID, &count); err != nil {
			return nil, err
		}
		stats[d1tvID] = count
	}

	return stats, rows.Err()
}
