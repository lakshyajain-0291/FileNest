package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"

	//database driver
	_ "github.com/mattn/go-sqlite3"
)

func Store(db_name string, peerID int, embedding []float64, ipv4 string, port int) {
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

	_, new_err := db.Exec("CREATE TABLE IF NOT EXISTS storage(peerID INTEGER PRIMARY KEY, embedding TEXT, ipv4 TEXT, port INTEGER)")
	if new_err != nil {
		fmt.Println(new_err) // Will only error for actual SQL problems, not existing tables
		log.Fatal(new_err)
	}

	string_embedding, err := json.Marshal(embedding)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	_, new_err2 := db.Exec("INSERT INTO storage VALUES(?, ?, ?, ?)", peerID, string(string_embedding), ipv4, port)
	if new_err2 != nil {
		fmt.Println("Error:", new_err2)
		log.Fatal(new_err2)
	}

}
