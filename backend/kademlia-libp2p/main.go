package main

import (
	"bufio"
	"context"
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
	interactive := flag.Bool("interactive", true, "Enable interactive CLI mode")
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

				// Trigger Kademlia protocol exchange to add to routing table
				go func(peerID peer.ID) {
					time.Sleep(2 * time.Second) // Wait for connection to stabilize
					err := node.Ping(peerID)
					if err != nil {
						log.Printf("Initial ping to bootstrap peer failed: %v", err)
					}
				}(peerInfo.ID)
			}
		}

		if connectedCount == 0 {
			log.Println("Warning: Failed to connect to any bootstrap peers")
		} else {
			fmt.Printf("Connected to %d bootstrap peer(s)\n", connectedCount)
		}
	}

	// Print status information
	fmt.Printf("\n=== Node Status ===\n")
	fmt.Printf("Host Peer ID: %s\n", host.ID())
	fmt.Printf("Depth Level: %d\n", *depth)
	fmt.Printf("Message Type: %s\n", *msgType)
	fmt.Printf("Bootstrap Peers: %s\n", *bootstrap)
	fmt.Printf("Auto-started: %t\n", *autoStart)
	fmt.Printf("Interactive Mode: %t\n", *interactive)

	if *depth < 4 {
		fmt.Printf("Node Type: Intermediate (D%d) - Routes and processes embeddings\n", *depth)
	} else {
		fmt.Printf("Node Type: Leaf (D4) - Final storage and retrieval\n")
	}

	// Create cross-platform quit channel
	quitCh := make(chan bool, 1)

	// Interactive CLI with proper ping functionality
	if *interactive {
		go func() {
			time.Sleep(2 * time.Second) // Wait for startup
			fmt.Printf("\n=== Interactive Commands ===\n")
			fmt.Printf("Available commands: store, get, ping, peers, contacts, dbstats, storage, info, help, quit\n")

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
				command := parts[0]

				switch command {
				case "store":
					if len(parts) >= 3 {
						key := parts[1]
						value := strings.Join(parts[2:], " ")
						start := time.Now()
						err := node.Store([]byte(key), []byte(value))
						duration := time.Since(start)
						if err != nil {
							fmt.Printf("Error storing: %v\n", err)
						} else {
							fmt.Printf("Successfully stored: %s = %s (took %v)\n", key, value, duration)
						}
					} else {
						fmt.Println("Usage: store <key> <value>")
					}

				case "get":
					if len(parts) >= 2 {
						key := parts[1]
						start := time.Now()
						value, err := node.FindValue([]byte(key))
						duration := time.Since(start)
						if err != nil {
							fmt.Printf("Error finding value: %v (took %v)\n", err, duration)
						} else {
							fmt.Printf("Found: %s = %s (took %v)\n", key, string(value), duration)
						}
					} else {
						fmt.Println("Usage: get <key>")
					}

				case "ping":
					if len(parts) >= 2 {
						peerIDStr := parts[1]
						peerID, err := peer.Decode(peerIDStr)
						if err != nil {
							fmt.Printf("Invalid peer ID: %v\n", err)
						} else {
							start := time.Now()

							// Use Kademlia PING protocol
							err := node.Ping(peerID)
							duration := time.Since(start)

							if err != nil {
								fmt.Printf("PING %s: failed - %v (took %v)\n", peerIDStr, err, duration)
							} else {
								fmt.Printf("PING %s: PONG received (took %v)\n", peerIDStr, duration)

								// Show that peer was added to routing table
								time.Sleep(100 * time.Millisecond) // Brief delay for routing table update
								contacts := node.GetRoutingTable().GetAllContacts()
								found := false
								for _, contact := range contacts {
									if contact.ID == peerID {
										found = true
										break
									}
								}

								if found {
									fmt.Printf("Peer %s added to routing table\n", peerIDStr)
								}
							}
						}
					} else {
						fmt.Println("Usage: ping <peer_id>")
					}

				case "peers":
					connectedPeers := host.Network().Peers()
					fmt.Printf("Connected peers: %d\n", len(connectedPeers))
					for i, peerID := range connectedPeers {
						peerInfo := host.Peerstore().PeerInfo(peerID)
						fmt.Printf("  %d. %s (%d addrs)\n", i+1, peerID, len(peerInfo.Addrs))
					}

				case "contacts":
					contacts := node.GetRoutingTable().GetAllContacts()
					fmt.Printf("Routing table contacts: %d\n", len(contacts))
					for i, contact := range contacts {
						fmt.Printf("  %d. ID: %s\n", i+1, contact.ID)
						fmt.Printf("      NodeID: %x\n", contact.NodeID[:min(8, len(contact.NodeID))])
						fmt.Printf("      Addrs: %v\n", contact.Addrs)
						fmt.Printf("      LastSeen: %v\n", contact.LastSeen)
						fmt.Println()
					}

				case "dbstats":
					connectedPeers := host.Network().Peers()
					contacts := node.GetRoutingTable().GetAllContacts()

					fmt.Printf("=== Database Statistics ===\n")
					fmt.Printf("Connected Peers: %d\n", len(connectedPeers))
					fmt.Printf("Routing Table Contacts: %d\n", len(contacts))
					fmt.Printf("Host Peer ID: %s\n", host.ID())
					fmt.Printf("Node Depth: %d\n", *depth)
					fmt.Printf("Message Type: %s\n", *msgType)
					fmt.Printf("Data Directory: %s\n", *dataDir)

				case "storage":
					fmt.Printf("=== Stored Data ===\n")
					fmt.Printf("Data Directory: %s\n", *dataDir)
					fmt.Printf("Use 'get <key>' to retrieve specific stored values\n")

				case "info", "status":
					fmt.Printf("=== Node Information ===\n")
					fmt.Printf("Peer ID: %s\n", host.ID())
					fmt.Printf("Depth: %d\n", *depth)
					fmt.Printf("Type: %s\n", *msgType)
					fmt.Printf("Data Directory: %s\n", *dataDir)
					fmt.Printf("Connected Peers: %d\n", len(host.Network().Peers()))
					fmt.Printf("Routing Contacts: %d\n", len(node.GetRoutingTable().GetAllContacts()))
					fmt.Printf("Auto-started: %t\n", *autoStart)
					fmt.Printf("Interactive Mode: %t\n", *interactive)

					fmt.Printf("\nListening Addresses:\n")
					for _, addr := range host.Addrs() {
						fmt.Printf("  %s/p2p/%s\n", addr, host.ID())
					}

				case "help":
					fmt.Printf("=== Available Commands ===\n")
					fmt.Printf("store <key> <value>  - Store a key-value pair in the DHT\n")
					fmt.Printf("get <key>            - Retrieve a value from the DHT\n")
					fmt.Printf("ping <peer_id>       - Send Kademlia PING to a peer\n")
					fmt.Printf("peers                - Show connected libp2p peers\n")
					fmt.Printf("contacts             - Show Kademlia routing table contacts\n")
					fmt.Printf("dbstats              - Show database and network statistics\n")
					fmt.Printf("storage              - Show information about stored data\n")
					fmt.Printf("info/status          - Show detailed node information\n")
					fmt.Printf("help                 - Show this help message\n")
					fmt.Printf("quit/exit            - Shutdown the node\n")

				case "quit", "exit":
					fmt.Println("Shutting down...")
					quitCh <- true
					return

				default:
					fmt.Printf("Unknown command: %s\n", command)
					fmt.Printf("Type 'help' for available commands\n")
				}
			}
		}()
	}

	fmt.Printf("\nNode is running...")
	if *interactive {
		fmt.Printf(" Type commands or Press Ctrl+C to shutdown\n")
	} else {
		fmt.Printf(" Press Ctrl+C to shutdown\n")
	}
	fmt.Printf("=====================================\n\n")

	// Graceful shutdown handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Wait for either OS signal or quit command
	select {
	case <-sigCh:
		fmt.Printf("\n=== Received shutdown signal ===\n")
	case <-quitCh:
		fmt.Printf("\n=== Received quit command ===\n")
	}

	fmt.Printf("=== Shutting Down ===\n")
	fmt.Println("Kademlia node stopping...")
	fmt.Println("Shutdown complete")
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
