package kademlia

import (
    "crypto/rand"
    "crypto/sha1"
    "encoding/hex"
    "fmt"
    "os"
    "path/filepath"
    "github.com/libp2p/go-libp2p/core/peer"
)

// NodeIDManager handles persistent node ID storage
type NodeIDManager struct {
    nodeID   []byte
    filePath string
}

// NewNodeIDManager creates or loads a persistent node ID
func NewNodeIDManager(dataDir string, peerID peer.ID) (*NodeIDManager, error) {
    if err := os.MkdirAll(dataDir, 0700); err != nil {
        return nil, fmt.Errorf("failed to create data directory: %w", err)
    }
    
    manager := &NodeIDManager{
        filePath: filepath.Join(dataDir, "node_id"),
    }
    
    if err := manager.loadOrGenerate(peerID); err != nil {
        return nil, err
    }
    
    return manager, nil
}

func (nim *NodeIDManager) loadOrGenerate(peerID peer.ID) error {
    // Try to load existing
    if data, err := os.ReadFile(nim.filePath); err == nil {
        nodeID, err := hex.DecodeString(string(data))
        if err == nil && len(nodeID) == KeySize {
            nim.nodeID = nodeID
            fmt.Printf("Loaded persistent node ID: %s\n", hex.EncodeToString(nodeID)[:16]+"...")
            return nil
        }
        fmt.Println("Invalid node ID file, generating new one")
    }
    
    // Generate new secure node ID
    nodeID := make([]byte, KeySize)
    if _, err := rand.Read(nodeID); err != nil {
        // Fallback to deterministic generation
        hash := sha1.Sum([]byte(peerID))
        copy(nodeID, hash[:])
        fmt.Println("Warning: Using deterministic node ID generation")
    }
    
    nim.nodeID = nodeID
    
    // Save to file
    nodeIDHex := hex.EncodeToString(nodeID)
    if err := os.WriteFile(nim.filePath, []byte(nodeIDHex), 0600); err != nil {
        return fmt.Errorf("failed to save node ID: %w", err)
    }
    
    fmt.Printf("Generated new node ID: %s\n", nodeIDHex[:16]+"...")
    return nil
}

func (nim *NodeIDManager) GetNodeID() []byte {
    result := make([]byte, len(nim.nodeID))
    copy(result, nim.nodeID)
    return result
}

func (nim *NodeIDManager) GetNodeIDString() string {
    return hex.EncodeToString(nim.nodeID)
}

// GenerateNodeID creates a deterministic node ID from peer ID (for temporary use)
func GenerateNodeID(peerID peer.ID) []byte {
    hash := sha1.Sum([]byte(peerID))
    return hash[:]
}
