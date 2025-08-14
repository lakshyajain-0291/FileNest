package identity

import (
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
)

const (
	NodeIDBits  = 160
	NodeIDBytes = NodeIDBits / 8
	defaultFile = "nodeid.bin"
)

// LoadOrCreateNodeID loads the NodeID from file, or generates and saves a new one if not present.
func LoadOrCreateNodeID(filePath string) ([]byte, error) {
	if filePath == "" {
		filePath = defaultFile
	}

	// Ensure directory exists
	dir := filepath.Dir(filePath)
	if dir != "." {
		if err := os.MkdirAll(dir, 0700); err != nil {
			return nil, fmt.Errorf("failed to create nodeid directory: %w", err)
		}
	}

	// Try reading existing NodeID
	if _, err := os.Stat(filePath); err == nil {
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("failed to read nodeid: %w", err)
		}
		if len(data) != NodeIDBytes {
			return nil, fmt.Errorf("invalid nodeid length: expected %d bytes, got %d", NodeIDBytes, len(data))
		}
		return data, nil
	}

	// Generate new NodeID
	nodeID := make([]byte, NodeIDBytes)
	if _, err := rand.Read(nodeID); err != nil {
		return nil, fmt.Errorf("failed to generate nodeid: %w", err)
	}

	// Save to file
	if err := os.WriteFile(filePath, nodeID, 0600); err != nil {
		return nil, fmt.Errorf("failed to save nodeid: %w", err)
	}

	return nodeID, nil
}
