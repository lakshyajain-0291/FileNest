package parser

import (
	"encoding/json"
	"os"
)

type ClusterFileMetadata struct {
	CreatedAt    string  `json:"created_at"`
	LastModified string  `json:"last_modified"`
	FileSize     float64 `json:"file_size"`
}

type ClusterFile struct {
	Filename  string              `json:"filename"`
	Metadata  ClusterFileMetadata `json:"metadata"`
	Embedding []float64           `json:"embedding"`
}

type Cluster struct {
	Centroid []float64     `json:"centroid"`
	Files    []ClusterFile `json:"files"`
}

type ClusterWrapper struct {
	Clusters []Cluster `json:"Clusters"`
}

// ParseCluster parses the JSON file at the given path and returns a slice of Cluster structs.
func ParseCluster(path string) ([]Cluster, error) {
	var wrapper ClusterWrapper
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
