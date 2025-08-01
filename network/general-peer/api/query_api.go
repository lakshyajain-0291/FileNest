package api

import (
	"encoding/json"
	"fmt"
	"general-peer/pkg/models"
	"net"
	"net/http"
	"strconv"
)

// ForwardQueryHandler handles POST requests to forward a query embedding to a list of peers via TCP.
func ForwardQueryHandler(w http.ResponseWriter, r *http.Request) {
	// Decode the request body into the request struct
	var req models.ForwardQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Marshal the query into JSON format
	data, err := json.Marshal(req.Query)
	if err != nil {
		http.Error(w, "Failed to marshal query: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Add newline as a delimiter (if the peer expects a delimiter)
	data = append(data, '\n')

	// Send the query to each peer individually via TCP
	for _, peer := range req.Peers {
		address := net.JoinHostPort(peer.IP, strconv.Itoa(peer.Port))

		conn, err := net.Dial("tcp", address)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to connect to peer %d (%s): %v", peer.ID, address, err), http.StatusInternalServerError)
			return
		}

		_, err = conn.Write(data)
		conn.Close()

		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to send to peer %d (%s): %v", peer.ID, address, err), http.StatusInternalServerError)
			return
		}
	}

	// Return success if all forwards succeeded
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "success",
	})
}
