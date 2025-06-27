package PostgreSQL

import (
    "context"
    "fmt"
    "time"

    "github.com/jackc/pgx/v5/pgxpool"
)

func ConnectDB(ctx context.Context, username string, pass string, db_name string) (*pgxpool.Pool, error) {
    connStr := fmt.Sprintf("postgres://%s:%s@localhost:5432/%s?sslmode=disable&pool_max_conns=10", username, pass, db_name)
    
    config, err := pgxpool.ParseConfig(connStr)
    if err != nil {
        return nil, fmt.Errorf("failed to parse config: %w", err)
    }

    pool, err := pgxpool.NewWithConfig(ctx, config)
    if err != nil {
        return nil, fmt.Errorf("failed to create connection pool: %w", err)
    }
    
    // Test the connection
    if err := pool.Ping(ctx); err != nil {
        pool.Close()
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }
    
    return pool, nil
}

func CreateTableIfNotExists(ctx context.Context, pool *pgxpool.Pool, table string) error {
    // Drop table if it exists to ensure clean schema
    dropSQL := fmt.Sprintf(`DROP TABLE IF EXISTS %s`, table)
    _, err := pool.Exec(ctx, dropSQL)
    if err != nil {
        return fmt.Errorf("failed to drop existing table: %w", err)
    }

    // Create table with correct schema
    sql := fmt.Sprintf(`
        CREATE TABLE %s (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            filepath TEXT NOT NULL,
            embedding FLOAT8[] NOT NULL,
            d1tv_id INTEGER NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
        )`, table)
    
    _, err = pool.Exec(ctx, sql)
    if err != nil {
        return fmt.Errorf("failed to create table: %w", err)
    }
    
    fmt.Printf("[INFO] Table '%s' created successfully\n", table)
    return nil
}

func AddData(ctx context.Context, pool *pgxpool.Pool, table string, filename string, filepath string, embedding []float64, d1tv_id int) error {
    sql := fmt.Sprintf(`INSERT INTO %s (filename, filepath, embedding, d1tv_id, created_at)
        VALUES ($1, $2, $3, $4, $5)`, table)

    ct, err := pool.Exec(ctx, sql,
        filename,
        filepath,
        embedding,
        d1tv_id,
        time.Now(),
    )
    
    if err != nil {
        return fmt.Errorf("database insert failed: %w", err)
    }
    
    fmt.Printf("[DB INSERT SUCCESS] Rows affected: %d\n", ct.RowsAffected())
    return nil
}
