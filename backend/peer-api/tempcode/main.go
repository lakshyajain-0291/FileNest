package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/centauri1219/Filenest/backend/p2p-api/tempcode/usecase"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/joho/godotenv"
)

var db *sql.DB

func init() {
	//load .env file
	err := godotenv.Load()
	if err != nil {
		log.Fatal("err loading .env file", err)
	}
	log.Println("loaded successfully ")

	// Connect to PostgreSQL
	dburi := os.Getenv("POSTGRES_URI")
	db, err = sql.Open("pgx", dburi) // special psql driver for golang
	if err != nil {
		log.Fatal("error connecting to PostgreS", err)
	}
	if err = db.Ping(); err != nil {
		log.Fatal("Error pinging PostgreSQL", err)
	}
	log.Println("PostgreSQL connected successfully")
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all connections, adjust for production
	},
}

func wsHandler(fileservice usecase.FileService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Println("WebSocket upgrade error:", err)
			return
		}
		defer conn.Close()

		// Extract peer info from query params (no logging)
		// peerID := r.URL.Query().Get("peer_id")
		// name := r.URL.Query().Get("name")
		// host, _, err := net.SplitHostPort(r.RemoteAddr)
		// ip := host
		// if err != nil {
		//     ip = r.RemoteAddr // fallback, should not happen
		// }
		// if peerID != "" {
		//     if err := usecase.LogPeerConnection(fileservice.DB, peerID, name, ip); err != nil {
		//         log.Println("Peer log error:", err)
		//     }
		// }

		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				log.Println("webSocket read error:", err)
				break
			}
			log.Printf("Received: %s", msg)

			// Parse the incoming message as JSON
			var req struct {
				Source       string    `json:"source"`
				SourceID     int       `json:"source_id"`
				Embed        []float64 `json:"embed"`
				PrevDepth    int       `json:"prev_depth"`
				QueryType    string    `json:"query_type"`
				Threshold    float64   `json:"threshold"`
				ResultsCount int       `json:"results_count"`
			}
			if err := json.Unmarshal(msg, &req); err != nil {
				log.Println("Invalid JSON or format:", err)
				conn.WriteMessage(websocket.TextMessage, []byte(`{"error":"invalid request format"}`))
				continue
			}

			// Pass the parsed request to the file service (update HandleWSMessage to accept this if needed)
			resp, err := fileservice.HandleWSMessage(msg) // optionally: pass req if you update the method
			if err != nil {
				log.Println("WebSocket handle error:", err)
				conn.WriteMessage(websocket.TextMessage, []byte(`{"error":"internal server error"}`))
				continue
			}
			err = conn.WriteMessage(websocket.TextMessage, resp)
			if err != nil {
				log.Println("WebSocket write error:", err)
				break
			}
		}

		// No peer disconnect logging
	}
}

func main() {
	defer db.Close()
	//create file service
	fileservice := usecase.FileService{DB: db} // pass db instead of MongoCollection

	//create router
	r := mux.NewRouter()                                           //directs incoming http requests to the handler funcs
	r.HandleFunc("/health", healthHandler).Methods(http.MethodGet) //check if server is running
	// Add WebSocket endpoint
	r.HandleFunc("/ws", wsHandler(fileservice))

	log.Println("Server is running on port 4444")
	http.ListenAndServe(":4444", r) // start the server on port 4444
}

func healthHandler(w http.ResponseWriter, r *http.Request) { //mapped to /health endpoint
	w.WriteHeader(http.StatusOK)   //http response
	w.Write([]byte("running....")) //write response to client
}
