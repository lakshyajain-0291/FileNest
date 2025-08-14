package kademlia

import "kademlia/pkg/storage"

type KademliaNode struct {
    NodeID           []byte              // Persistent NodeID
    PeerID       string              // Ephemeral libp2p PeerID
    routingTable *RoutingTable
    storage      storage.Interface   // Add this field
    network      NetworkInterface
}

func NewKademliaNode(nodeID []byte, peerID string, address string, network NetworkInterface, dbPath string) (*KademliaNode, error) {
    // Initialize SQLite storage
    sqliteStorage, err := storage.NewSQLiteStorage(dbPath)
    if err != nil {
        return nil, err
    }
    
    return &KademliaNode{
        NodeID:           nodeID,
        PeerID:       peerID,
        routingTable: NewRoutingTable(nodeID, peerID, 20), // K=20
        storage:      sqliteStorage,  // Initialize storage
        network:      network,
    }, nil
}


func (k *KademliaNode) GetID() []byte {
	return k.NodeID
}

func (k *KademliaNode) GetAddress() string {
	return k.PeerID
}

