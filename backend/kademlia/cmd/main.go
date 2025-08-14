package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"kademlia/pkg/db"
	"kademlia/pkg/helpers"
	"kademlia/pkg/identity"
	"kademlia/pkg/libp2p" // <-- your bootstrap package
	routingtable "kademlia/pkg/routingTable"
	"kademlia/pkg/types"
)

func main() {
	// Step 1: Bootstrap decision
	ctx := context.Background()
	role, err := libp2p.Bootstrap(ctx, libp2p.BootstrapConfig{
		BootstrapAddr: "",              
		Timeout:       5 * time.Second,
	}, func(addr string) error {
		// Temporary dialFn (replace with libp2p dial later)
		fmt.Printf("Dialing %s...\n", addr)
		// Simulate dial success/failure
		return fmt.Errorf("simulated dial failure")
	})
	if err != nil {
		log.Fatalf("bootstrap failed: %v", err)
	}

	fmt.Println("Node role:", role)

	// Step 2: Load or create persistent NodeID
	selfNodeID, err := identity.LoadOrCreateNodeID("data/nodeid.bin")
	if err != nil {
		log.Fatalf("failed to load/create node ID: %v", err)
	}

	// Temporary PeerID for this session
	selfPeerID := "peer_self_test"
	fmt.Printf("Self NodeID: %x\nSelf PeerID: %s\n", selfNodeID, selfPeerID)

	// Step 3: Init SQLite DB
	database, err := db.NewDB("test_rt3.db")
	if err != nil {
		log.Fatal(err)
	}

	// Step 4: Create a routing table
	rt := routingtable.NewRoutingTable(selfNodeID, selfPeerID, 3)

	// Step 5: Add fake peers and persist them
	for i := range 5 {
		p := types.PeerInfo{
			NodeID: helpers.RandomNodeID(),
			PeerID: fmt.Sprintf("peer%d", i),
		}
		idx := helpers.BucketIndex(selfNodeID, p.NodeID)
		rt.Update(p) // Instead of AddPeer, use your routing table's Update
		if err := database.SavePeer(idx, p); err != nil {
			log.Fatal(err)
		}
	}
	fmt.Println("Original table:", rt.Buckets)

	// Step 6: Load into a new routing table instance
	rt2 := routingtable.NewRoutingTable(selfNodeID, selfPeerID, 3)
	if err := database.LoadRoutingTable(rt2); err != nil {
		log.Fatal(err)
	}

	fmt.Println("Loaded table:", rt2.Buckets)
}
