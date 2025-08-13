package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/peer"
	ws "github.com/libp2p/go-libp2p/p2p/transport/websocket"
	"github.com/multiformats/go-multiaddr"

	"kademlia-libp2p/integration"
	"kademlia-libp2p/kademlia"
)

func main() {
	// Command line flags
	port := flag.Int("port", 0, "Port to listen on (0 for random)")
	bootstrap := flag.String("bootstrap", "", "Bootstrap peer addresses (comma separated)")
	dataDir := flag.String("datadir", "./kademlia_data", "Data directory for persistent storage")
	depth := flag.Int("depth", 2, "Node depth in embedding hierarchy (1-4)")
	msgType := flag.String("type", "search", "Message type (search/store)")
	autoStart := flag.Bool("autostart", true, "Automatically start the node")
	flag.Parse()

	fmt.Printf("=== Kademlia Node with Embedding Integration ===\n")
	fmt.Printf("Starting node at depth %d, handling %s operations\n", *depth, *msgType)
	fmt.Printf("Data Directory: %s\n", *dataDir)

	// Create libp2p host with WebSocket transport
	host, err := libp2p.New(
		libp2p.ListenAddrStrings(fmt.Sprintf("/ip4/0.0.0.0/tcp/%d/ws", *port)),
		libp2p.Transport(ws.New),
	)
	if err != nil {
		panic(fmt.Sprintf("Failed to create libp2p host: %v", err))
	}
	defer host.Close()

	fmt.Printf("Peer ID: %s\n", host.ID())
	fmt.Printf("Listening on:\n")
	for _, addr := range host.Addrs() {
		fmt.Printf("  %s/p2p/%s\n", addr, host.ID())
	}
	fmt.Println()

	// Create Kademlia node
	node, err := kademlia.NewNode(host, *dataDir)
	if err != nil {
		panic(fmt.Sprintf("Failed to create Kademlia node: %v", err))
	}

	// Start Kademlia node if autostart is enabled
	if *autoStart {
		if err := node.Start(); err != nil {
			panic(fmt.Sprintf("Failed to start node: %v", err))
		}
		fmt.Println("Kademlia node started successfully")
	}

	// Bootstrap connections if provided
	if *bootstrap != "" {
		fmt.Printf("Attempting to bootstrap with: %s\n", *bootstrap)

		bootstrapAddrs := strings.Split(*bootstrap, ",")
		connectedCount := 0

		for _, addrStr := range bootstrapAddrs {
			addrStr = strings.TrimSpace(addrStr)
			if addrStr == "" {
				continue
			}

			maddr, err := multiaddr.NewMultiaddr(addrStr)
			if err != nil {
				log.Printf("Invalid multiaddr %s: %v", addrStr, err)
				continue
			}

			peerInfo, err := peer.AddrInfoFromP2pAddr(maddr)
			if err != nil {
				log.Printf("Invalid peer info from %s: %v", addrStr, err)
				continue
			}

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			err = host.Connect(ctx, *peerInfo)
			cancel()

			if err != nil {
				log.Printf("Failed to connect to bootstrap peer %s: %v", peerInfo.ID, err)
			} else {
				fmt.Printf("Successfully connected to bootstrap peer: %s\n", peerInfo.ID)
				connectedCount++

				// Add to routing table
				node.AddBootstrapPeer(peerInfo.ID, peerInfo.Addrs)
			}
		}

		if connectedCount == 0 {
			log.Println("Warning: Failed to connect to any bootstrap peers")
		} else {
			fmt.Printf("Connected to %d bootstrap peer(s)\n", connectedCount)
		}
	}

	// Create integration bridge
	bridge, err := integration.NewKademliaNetworkBridge(node, *depth, *msgType)
	if err != nil {
		log.Fatal("Failed to create integration bridge:", err)
	}

	fmt.Printf("Integration bridge created successfully\n")
	fmt.Printf("Host Peer ID for relay layer: %s\n", bridge.GetHostPeerID())

	// Set up network message handler for the bridge
	handleNetworkMessage := func(msgBytes []byte) error {
		var networkMsg integration.NetworkMessage
		if err := json.Unmarshal(msgBytes, &networkMsg); err != nil {
			return fmt.Errorf("failed to parse network message: %w", err)
		}

		if err := bridge.ProcessNetworkMessage(networkMsg); err != nil {
			return fmt.Errorf("failed to process message: %w", err)
		}

		return nil
	}

	// Example: Start a goroutine to handle incoming network messages
	// (In your actual implementation, this would be called by your network layer)
	go func() {
		log.Println("Network message handler ready...")

		// Example of how your network layer would send messages
		time.Sleep(2 * time.Second) // Wait for startup

		// Example embedding search message
		exampleSearchMsg := integration.NetworkMessage{
			Type: "embedding_search",
			Data: json.RawMessage(`{
                "source": "example_client",
                "source_id": 1,
                "embed": [0.1, 0.2, 0.3, 0.4, 0.5],
                "prev_depth": 0,
                "query_type": "search",
                "threshold": 0.8,
                "results_count": 10,
                "target_node_id": 123
            }`),
			Source:    host.ID(),
			Timestamp: time.Now().Unix(),
		}

		msgBytes, _ := json.Marshal(exampleSearchMsg)
		if err := handleNetworkMessage(msgBytes); err != nil {
			log.Printf("Example message failed: %v", err)
		} else {
			log.Println("Example embedding search message processed")
		}
	}()

	// Print status information
	fmt.Printf("\n=== Node Status ===\n")
	fmt.Printf("Kademlia Node ID: %s\n", node.GetNodeIDString())
	fmt.Printf("Depth Level: %d\n", *depth)
	fmt.Printf("Message Type: %s\n", *msgType)
	fmt.Printf("Bootstrap Peers: %s\n", *bootstrap)
	fmt.Printf("Auto-started: %t\n", *autoStart)

	if *depth < 4 {
		fmt.Printf("Node Type: Intermediate (D%d) - Routes and processes embeddings\n", *depth)
	} else {
		fmt.Printf("Node Type: Leaf (D4) - Final storage and retrieval\n")
	}

	fmt.Printf("\n=== Available Operations ===\n")
	fmt.Printf("- Embedding Search: Routes through Kademlia to find target nodes\n")
	fmt.Printf("- Embedding Store: Stores embeddings with centroid updates\n")
	fmt.Printf("- Kademlia Find: Standard DHT key-value lookups\n")
	fmt.Printf("- Peer Discovery: Automatic network topology maintenance\n")

	fmt.Printf("\n=== Ready for Network Messages ===\n")
	fmt.Printf("The node is now ready to receive messages from your network layer.\n")
	fmt.Printf("Integration bridge will handle routing through Kademlia DHT.\n")
	fmt.Printf("Host peer ID (%s) should be registered with your relay layer.\n", bridge.GetHostPeerID())

	// Graceful shutdown handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	fmt.Printf("\nNode is running... Press Ctrl+C to shutdown\n")
	fmt.Printf("=====================================\n\n")

	// Wait for shutdown signal
	<-sigCh

	fmt.Printf("\n=== Shutting Down ===\n")

	// Cleanup
	if err := node.Stop(); err != nil {
		log.Printf("Error stopping Kademlia node: %v", err)
	} else {
		fmt.Println("Kademlia node stopped")
	}

	fmt.Println("Host connection closed")
	fmt.Println("Shutdown complete")
}
