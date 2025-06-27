package main

import (
    "XSproto/TV"
    "XSproto/PostgreSQL"
    "context"
    "fmt"
    "io/fs"
    "log"
    "math"
    "math/rand"
    "os"
    "os/signal"
    "path/filepath"
    "strings"
    "sync"
    "syscall"
    "time"

    "github.com/jackc/pgx/v5/pgxpool" 
)


func generate_sample_embedding(content string, n int) []float64 {
    embedding := make([]float64, 0, n)
    source := rand.NewSource(time.Now().UnixNano())
    r := rand.New(source)
    for i := 0; i < n; i++ {
        embedding = append(embedding, r.Float64())
    }
    return embedding
}

func cosineSimilarity(a, b []float64) float64 {
    if len(a) != len(b) {
        log.Printf("Vector lengths do not match: %d vs %d", len(a), len(b))
        return 0
    }
    var dotProduct, normA, normB float64
    for i := range a {
        dotProduct += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    if normA == 0 || normB == 0 {
        return 0
    }
    return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func logging(data string) {
    f, err := os.OpenFile("logs.txt", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
    if err != nil {
        log.Printf("Failed to open log file: %v", err)
        return
    }
    defer f.Close()

    _, err = f.Write([]byte(data + "\n"))
    if err != nil {
        log.Printf("Failed to write to log file: %v", err)
    }
}

func worker(ctx context.Context, id int, filejobs <-chan string, pool *pgxpool.Pool, table string, wg *sync.WaitGroup) {
    defer wg.Done()
    
    for {
        select {
        case <-ctx.Done():
            fmt.Printf("[Worker-%d] Context cancelled, exiting...\n", id)
            return
        case j, ok := <-filejobs:
            if !ok {
                fmt.Printf("[Worker-%d] Job channel closed, exiting...\n", id)
                return
            }
            
            data, err := os.ReadFile(j)
            if err != nil {
                fmt.Printf("[Worker-%d] Read error: %v\n", id, err)
                continue
            }
            Data := string(data)

            taggingVectors := TV.Generatetaggingvectors()
            embedding := generate_sample_embedding(Data, 128)

            var maxIndex int
            var maxSim float64 = -1

            for i, tagVec := range taggingVectors {
                tagVecFloat := make([]float64, len(tagVec))
                for k, v := range tagVec {
                    tagVecFloat[k] = float64(v)
                }
                sim := cosineSimilarity(embedding, tagVecFloat)
                if sim > maxSim {
                    maxSim = sim
                    maxIndex = i
                }
            }
            
            parts := strings.Split(j, string(filepath.Separator))
            filename := parts[len(parts)-1]

            fileCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
            new_err := PostgreSQL.AddData(fileCtx, pool, table, filename, j, embedding, maxIndex)
            cancel()
            
            if new_err != nil {
                fmt.Printf("[ERROR] Failed to insert %s: %v\n", filename, new_err)
            } else {
                fmt.Printf("[INFO] Inserted %s successfully.\n", filename)
            }

            logMsg := fmt.Sprintf("[Worker-%d] Processed file: %s → D1TV: %d (similarity: %f)", id, j, maxIndex, maxSim)
            logging(logMsg)
        }
    }
}

func Assignjobs(filejobs chan<- string, root string) {
    defer close(filejobs)

    err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
        if err != nil {
            fmt.Printf("Error accessing path %q: %v\n", path, err)
            return err
        }
        if !d.IsDir() {
            filejobs <- path
        }
        return nil
    })

    if err != nil {
        fmt.Printf("Error walking the directory: %v\n", err)
    }
}

func main() {
    buffer_size := 10
    root := "./TestFolder"
    jobs := make(chan string, buffer_size)
    var username string = "postgres"
    var pass string = "Ishat@123"
    var db_name string = "mydb"
    var table_name string = "filenest"

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    c := make(chan os.Signal, 1)
    signal.Notify(c, os.Interrupt, syscall.SIGTERM)

    go func() {
        <-c
        fmt.Println("\n[INFO] Shutdown signal received, shutting down gracefully...")
        cancel()
    }()

    // Use connection pool instead of single connection
    pool, err := PostgreSQL.ConnectDB(ctx, username, pass, db_name)
    if err != nil {
        log.Fatal(err)
    }
    defer pool.Close()

    // Create/recreate table with correct schema
    err = PostgreSQL.CreateTableIfNotExists(ctx, pool, table_name)
    if err != nil {
        log.Fatal(err)
    }

    var wg sync.WaitGroup
    numWorkers := 5

    for w := 1; w <= numWorkers; w++ {
        wg.Add(1)
        go worker(ctx, w, jobs, pool, table_name, &wg)
    }

    go Assignjobs(jobs, root)

    wg.Wait()
    fmt.Println("[INFO] All workers finished, shutting down...")
}
