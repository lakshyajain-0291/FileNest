package kademlia

import (
	"fmt"
	"kademlia/pkg/storage"
)

type KademliaNode struct {
	NodeID       []byte // Persistent NodeID
	PeerID       string // Ephemeral libp2p PeerID
	routingTable *RoutingTable
	storage      storage.Interface // Add this field
	network      NetworkInterface
}

func NewKademliaNode(nodeID []byte, peerID string, network NetworkInterface, dbPath string) (*KademliaNode, error) {
	// Initialize SQLite storage
	sqliteStorage, err := storage.NewSQLiteStorage(dbPath)
	if err != nil {
		return nil, err
	}

	return &KademliaNode{
		NodeID:       nodeID,
		PeerID:       peerID,
		routingTable: NewRoutingTable(nodeID, peerID, 20), // K=20
		storage:      sqliteStorage,                       // Initialize storage
		network:      network,
	}, nil
}

func (k *KademliaNode) GetID() []byte {
	return k.NodeID
}

func (k *KademliaNode) GetAddress() string {
	return k.PeerID
}

// PrintRoutingTable displays the node's routing table
func (k *KademliaNode) PrintRoutingTable() {
	fmt.Printf("\n=== ROUTING TABLE FOR NODE %x ===\n", k.NodeID[:8])
	k.routingTable.PrintRoutingTable()
}

// PrintRoutingTableSummary displays a compact view of the routing table
func (k *KademliaNode) PrintRoutingTableSummary() {
	fmt.Printf("\n=== SUMMARY FOR NODE %x ===\n", k.NodeID[:8])
	k.routingTable.PrintRoutingTableSummary()
}

// PrintPeerInfo displays detailed peer information
func (k *KademliaNode) PrintPeerInfo() {
	fmt.Printf("\n=== PEER INFO FOR NODE %x ===\n", k.NodeID[:8])
	k.routingTable.PrintPeerInfo()
}

// PrintBucket displays a specific bucket's contents
func (k *KademliaNode) PrintBucket(bucketIndex int) {
	fmt.Printf("\n=== BUCKET %d FOR NODE %x ===\n", bucketIndex, k.NodeID[:8])
	k.routingTable.PrintBucket(bucketIndex)
}
