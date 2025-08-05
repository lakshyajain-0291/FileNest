package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"

	"kademlia-libp2p/kademlia"
)

func main() {
	port := flag.Int("port", 0, "Port to listen on (0 for random)")
	bootstrap := flag.String("bootstrap", "", "Bootstrap peer addresses (comma separated)")
	dataDir := flag.String("datadir", "./kademlia_data", "Data directory for persistent storage")
	autoStart := flag.Bool("autostart", true, "Automatically start the node")
	flag.Parse()

	// Create libp2p host - REMOVED EnableAutoRelay()
	host, err := libp2p.New(
		libp2p.ListenAddrStrings(fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", *port)),
	)
	if err != nil {
		panic(fmt.Sprintf("Failed to create libp2p host: %v", err))
	}
	defer host.Close()

	fmt.Printf("=== Kademlia Node ===\n")
	fmt.Printf("Peer ID: %s\n", host.ID())
	fmt.Printf("Data Directory: %s\n", *dataDir)
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
	defer node.Stop()

	// Start the node if autostart is enabled
	if *autoStart {
		if err := node.Start(); err != nil {
			panic(fmt.Sprintf("Failed to start node: %v", err))
		}
	}

	// Handle bootstrap peers
	if *bootstrap != "" {
		fmt.Printf("Parsing bootstrap addresses...\n")
		var bootstrapPeers []peer.ID

		for _, addrStr := range strings.Split(*bootstrap, ",") {
			addrStr = strings.TrimSpace(addrStr)
			if addrStr == "" {
				continue
			}

			fmt.Printf("Processing bootstrap address: %s\n", addrStr)

			maddr, err := multiaddr.NewMultiaddr(addrStr)
			if err != nil {
				fmt.Printf("Invalid bootstrap address %s: %v\n", addrStr, err)
				continue
			}

			peerInfo, err := peer.AddrInfoFromP2pAddr(maddr)
			if err != nil {
				fmt.Printf("Failed to extract peer info from %s: %v\n", addrStr, err)
				continue
			}

			fmt.Printf("Extracted peer ID: %s\n", peerInfo.ID)
			bootstrapPeers = append(bootstrapPeers, peerInfo.ID)

			// Pre-connect to bootstrap peer
			fmt.Printf("Pre-connecting to bootstrap peer %s...\n", peerInfo.ID)
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			err = host.Connect(ctx, *peerInfo)
			cancel()

			if err != nil {
				fmt.Printf("Failed to pre-connect to bootstrap peer %s: %v\n", peerInfo.ID, err)
			} else {
				fmt.Printf("Pre-connected to bootstrap peer: %s\n", peerInfo.ID)
			}
		}

		if len(bootstrapPeers) > 0 && node.IsRunning() {
			fmt.Printf("Starting Kademlia bootstrap with %d peers...\n", len(bootstrapPeers))
			if err := node.Bootstrap(bootstrapPeers); err != nil {
				fmt.Printf("Bootstrap failed: %v\n", err)
			} else {
				fmt.Println("Bootstrap completed successfully!")
			}
		} else if len(bootstrapPeers) == 0 {
			fmt.Println("No valid bootstrap peers found")
		}
	}

	// Interactive CLI
	fmt.Println("=== Interactive CLI ===")
	fmt.Println("Commands:")
	fmt.Println("  start                    - Start the node")
	fmt.Println("  stop                     - Stop the node")
	fmt.Println("  store <key> <value>      - Store a key-value pair")
	fmt.Println("  get <key>                - Retrieve a value by key")
	fmt.Println("  ping <peer_id>           - Ping a specific peer")
	fmt.Println("  peers                    - Show connected peer count")
	fmt.Println("  contacts                 - Show all contacts")
	fmt.Println("  buckets                  - Show bucket information")
	fmt.Println("  storage                  - Show stored keys")
	fmt.Println("  info                     - Show node information")
	fmt.Println("  help                     - Show this help message")
	fmt.Println("  quit                     - Exit the program")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("kademlia> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		parts := strings.Fields(input)
		command := strings.ToLower(parts[0])

		switch command {
		case "start":
			if node.IsRunning() {
				fmt.Println("Node is already running")
			} else {
				if err := node.Start(); err != nil {
					fmt.Printf("Failed to start node: %v\n", err)
				} else {
					fmt.Println("Node started successfully")
				}
			}

		case "stop":
			if !node.IsRunning() {
				fmt.Println("Node is not running")
			} else {
				node.Stop()
				fmt.Println("Node stopped")
			}

		case "store":
			if !node.IsRunning() {
				fmt.Println("Node is not running. Use 'start' command first.")
				continue
			}
			if len(parts) < 3 {
				fmt.Println("Usage: store <key> <value>")
				continue
			}
			key := parts[1]
			value := strings.Join(parts[2:], " ")

			fmt.Printf("Storing key='%s' value='%s'...\n", key, value)
			err := node.Store([]byte(key), []byte(value))
			if err != nil {
				fmt.Printf("Store failed: %v\n", err)
			} else {
				fmt.Printf("Successfully stored: %s\n", key)
			}

		case "get":
			if !node.IsRunning() {
				fmt.Println("Node is not running. Use 'start' command first.")
				continue
			}
			if len(parts) < 2 {
				fmt.Println("Usage: get <key>")
				continue
			}
			key := parts[1]

			fmt.Printf("Looking up key='%s'...\n", key)
			value, err := node.FindValue([]byte(key))
			if err != nil {
				fmt.Printf("Get failed: %v\n", err)
			} else {
				fmt.Printf("Found: %s = %s\n", key, string(value))
			}

		case "ping":
			if !node.IsRunning() {
				fmt.Println("Node is not running. Use 'start' command first.")
				continue
			}
			if len(parts) < 2 {
				fmt.Println("Usage: ping <peer_id>")
				continue
			}
			peerIDStr := parts[1]

			peerID, err := peer.Decode(peerIDStr)
			if err != nil {
				fmt.Printf("Invalid peer ID: %v\n", err)
				continue
			}

			// Check if peer is in contacts
			contacts := node.GetAllContacts()
			found := false
			for _, contact := range contacts {
				if contact.ID == peerID {
					found = true
					fmt.Printf("Found peer in contacts: %s\n", contact.ID)
					break
				}
			}

			if !found {
				fmt.Printf("Peer %s not found in contacts\n", peerID)
				fmt.Println("Available contacts:")
				for i, contact := range contacts {
					fmt.Printf("  %d. %s\n", i+1, contact.ID)
				}
				continue
			}

			// Check libp2p connection
			conns := host.Network().ConnsToPeer(peerID)
			fmt.Printf("Active connections to peer: %d\n", len(conns))

			fmt.Printf("Pinging %s...\n", peerID)
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			err = node.Ping(ctx, peerID)
			cancel()

			if err != nil {
				fmt.Printf("Ping failed: %v\n", err)
			} else {
				fmt.Printf("Ping successful!\n")
			}

		case "peers":
			count := node.GetPeerCount()
			fmt.Printf("Connected peers: %d\n", count)

		case "contacts":
			contacts := node.GetAllContacts()
			if len(contacts) == 0 {
				fmt.Println("No contacts in routing table")
			} else {
				fmt.Printf("Contacts (%d total):\n", len(contacts))
				for i, contact := range contacts {
					fmt.Printf("  %d. %s\n", i+1, contact.String())
				}
			}

		case "buckets":
			buckets := node.GetBucketInfo()
			if len(buckets) == 0 {
				fmt.Println("No active buckets")
			} else {
				fmt.Printf("Active buckets (%d total):\n", len(buckets))
				for bucket, count := range buckets {
					fmt.Printf("  Bucket %d: %d contacts\n", bucket, count)
				}
			}

		case "storage":
			count, keys := node.GetStorageInfo()
			fmt.Printf("Stored keys: %d\n", count)
			if count > 0 {
				for i, key := range keys {
					fmt.Printf("  %d. %s\n", i+1, key)
				}
			}

		case "info":
			fmt.Printf("Node Information:\n")
			fmt.Printf("  Running: %v\n", node.IsRunning())
			fmt.Printf("  Peer ID: %s\n", host.ID())
			fmt.Printf("  Node ID: %s\n", node.GetNodeID())
			fmt.Printf("  Contacts: %d\n", node.GetPeerCount())

			count, _ := node.GetStorageInfo()
			fmt.Printf("  Stored Keys: %d\n", count)

			fmt.Printf("  Addresses:\n")
			for _, addr := range host.Addrs() {
				fmt.Printf("    %s/p2p/%s\n", addr, host.ID())
			}

		case "help":
			fmt.Println("Available commands:")
			fmt.Println("  start, stop, store <key> <value>, get <key>")
			fmt.Println("  ping <peer_id>, peers, contacts, buckets")
			fmt.Println("  storage, info, help, quit")

		case "quit", "exit":
			fmt.Println("Shutting down...")
			return

		case "debug":
			fmt.Printf("=== Debug Information ===\n")
			contacts := node.GetAllContacts()
			fmt.Printf("Total contacts: %d\n", len(contacts))

			for i, contact := range contacts {
				conns := host.Network().ConnsToPeer(contact.ID)
				fmt.Printf("%d. Peer: %s\n", i+1, contact.ID)
				fmt.Printf("   Addresses: %v\n", contact.Addrs)
				fmt.Printf("   Connections: %d\n", len(conns))
				fmt.Printf("   Last seen: %v\n", contact.LastSeen)
				fmt.Println()
			}

		default:
			// Check if it's a shortcut for common operations
			if len(parts) == 1 && isNumeric(parts[0]) {
				// Quick peer count check
				count := node.GetPeerCount()
				fmt.Printf("Peers: %d\n", count)
			} else {
				fmt.Printf("Unknown command: %s (type 'help' for available commands)\n", command)
			}
		}
	}
}

func isNumeric(s string) bool {
	_, err := strconv.Atoi(s)
	return err == nil
}
