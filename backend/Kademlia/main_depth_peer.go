package main

import (
	findvalue "dht/RPC/find_value"
	"dht/routing_table"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/p2p/transport/tcp"
)

// write a function to generate a peer id for a device. ensure that once generated the peer id doesn't change for that particular device
var localPeerID int

// create a global routing table variable which stores and retrives the records in local memory. 
var rt *routing_table.RoutingTable

func main() {
	host, err := libp2p.New(libp2p.Transport(tcp.NewTCPTransport))

	if err != nil {
		log.Fatal(err)
		// return err
	}
	defer host.Close()

	// Register a stream handler to intercept messages
	host.SetStreamHandler("/jsonmessages/1.0.0", func(s network.Stream) {
		findvalue.HandleJSONMessages(s, localPeerID, rt) // need to create a global routing table which is stored in memory
	})

	fmt.Printf("Host ID: %s\n", host.ID())
	fmt.Printf("Listening on: %v\n", host.Addrs())

	// Keep running
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
}
