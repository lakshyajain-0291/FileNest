package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"

	//database driver
	_ "github.com/mattn/go-sqlite3"
)


// Record struct to represent a row from the database
type Record struct {
    PeerID    int
    Embedding []float64
    Multiaddr string
}

func Store(db_name string, peerID int, embedding []float64, multiaddr string) {
	// Connect to the SQLite database
	db_name = fmt.Sprintf("./%s", db_name)
	db, err := sql.Open("sqlite", db_name)
	if err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}

	defer db.Close()
	fmt.Println("Connected to the SQLite database successfully.")

	// creates a table if it doesn't already exist

	_, new_err := db.Exec("CREATE TABLE IF NOT EXISTS storage(peerID INTEGER PRIMARY KEY, embedding TEXT, multiaddr TEXT)")
	if new_err != nil {
		fmt.Println(new_err) // Will only error for actual SQL problems, not existing tables
		log.Fatal(new_err)
	}

	string_embedding, err := json.Marshal(embedding)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	_, new_err2 := db.Exec("INSERT INTO storage VALUES(?, ?, ?)", peerID, string(string_embedding), multiaddr)
	if new_err2 != nil {
		fmt.Println("Error:", new_err2)
		log.Fatal(new_err2)
	}

}

func ReadAll(db_name string) ([]Record, error) {
    db_name = fmt.Sprintf("./%s", db_name)
    db, err := sql.Open("sqlite3", db_name)
    if err != nil {
        return nil, err
    }
    defer db.Close()

    rows, err := db.Query("SELECT peerID, embedding, multiaddr FROM storage")
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var results []Record
    for rows.Next() {
        var r Record
        var embeddingJSON string
        if err := rows.Scan(&r.PeerID, &embeddingJSON, &r.Multiaddr); err != nil {
            log.Printf("scan error: %v", err)
            continue
        }
        if err := json.Unmarshal([]byte(embeddingJSON), &r.Embedding); err != nil {
            log.Printf("json unmarshal error: %v", err)
            continue
        }
        results = append(results, r)
    }
    if err := rows.Err(); err != nil {
        return results, err
    }
    return results, nil
}

