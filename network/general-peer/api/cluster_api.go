package api

import (
	"encoding/json"
	"general-peer/pkg/models"
	"general-peer/pkg/parser"
	"net/http"
)

// ClusterHandler handles a POST request to parse a cluster file and return the result as JSON.
func ClusterHandler(w http.ResponseWriter, r *http.Request) {

	// Decode the request body into the ClusterRequest struct
	var req models.ClusterRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	// Parse the file at the provided path to extract cluster data
	clusters, err := parser.ParseCluster(req.Path)
	if err != nil {
		http.Error(w, "Parse error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Return the parsed cluster data as JSON
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(clusters)
}
