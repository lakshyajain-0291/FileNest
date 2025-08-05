package kademlia

import (
    "crypto/rand"
    "encoding/hex"
    "fmt"
    "os"
    "path/filepath"
    "github.com/libp2p/go-libp2p/core/peer"
)

// Get or create persistent node ID
func GetOrCreateNodeID(dataDir string, peerID peer.ID) ([]byte, error) {
    os.MkdirAll(dataDir, 0700)
    nodeIDFile := filepath.Join(dataDir, "node_id")
    
    // Try to load existing
    if data, err := os.ReadFile(nodeIDFile); err == nil {
        nodeID, err := hex.DecodeString(string(data))
        if err == nil && len(nodeID) == KeySize {
            fmt.Printf("Loaded node ID: %s\n", hex.EncodeToString(nodeID)[:16]+"...")
            return nodeID, nil
        }
    }
    
    // Generate new
    nodeID := make([]byte, KeySize)
    rand.Read(nodeID)
    
    // Save
    nodeIDHex := hex.EncodeToString(nodeID)
    err := os.WriteFile(nodeIDFile, []byte(nodeIDHex), 0600)
    if err != nil {
        return nil, err
    }
    
    fmt.Printf("Generated node ID: %s\n", nodeIDHex[:16]+"...")
    return nodeID, nil
}
