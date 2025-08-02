package parser

import (
	"encoding/json"
	"general-peer/pkg/models"
	"os"
)

// ParseCluster parses the JSON file at the given path and returns a slice of Cluster structs.
func ParseCluster(path string) ([]models.Cluster, error) {
	var wrapper models.ClusterWrapper
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	
	err = json.Unmarshal(data, &wrapper)
	if err != nil {
		return nil, err
	}
	return wrapper.Clusters, nil
}
