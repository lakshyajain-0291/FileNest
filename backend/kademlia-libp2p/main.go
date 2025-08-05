package main

import (
	"context"
	"flag"
	"fmt"
	"strings"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"

	kademlia "kademlia-libp2p/kademlia"
)

func main() {
	var (
		port      = flag.Int("port", 0, "Port to listen on (0 for random)")
		bootstrap = flag.String("bootstrap", "", "Bootstrap peer addresses (comma separated)")
	)
	flag.Parse()

	// Create libp2p host
	host, err := libp2p.New(
		libp2p.ListenAddrStrings(fmt.Sprintf("/ip4/0.0.0.0/tcp/ws/%d", *port)),
		libp2p.EnableRelay(),
	)
	if err != nil {
		panic(err)
	}
	defer host.Close()

	fmt.Printf("Host created with ID: %s\n", host.ID())
	fmt.Printf("Listening on:")
	for _, addr := range host.Addrs() {
		fmt.Printf("  %s/p2p/%s\n", addr, host.ID())
	}

	// Parse bootstrap peers
	var bootstrapPeers []peer.ID
	if *bootstrap != "" {
		for _, addrStr := range strings.Split(*bootstrap, ",") {
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

			bootstrapPeers = append(bootstrapPeers, peerInfo.ID)

			// Connect to bootstrap peer
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			err = host.Connect(ctx, *peerInfo)
			cancel()
			if err != nil {
				fmt.Printf("Failed to connect to bootstrap peer %s: %v\n", peerInfo.ID, err)
			} else {
				fmt.Printf("Connected to bootstrap peer: %s\n", peerInfo.ID)
			}
		}
	}

	// Create and start Kademlia node
	kademlia := kademlia.NewKademliaNode(host, bootstrapPeers)
	err = kademlia.Start()
	if err != nil {
		panic(err)
	}
	defer kademlia.Stop()

	// Interactive CLI
	fmt.Println("\nKademlia node started! Available commands:")
	fmt.Println("  store <key> <value> - Store a key-value pair")
	fmt.Println("  get <key> - Retrieve a value by key")
	fmt.Println("  peers - Show peer count")
	fmt.Println("  quit - Exit")

	var input string
	for {
		fmt.Print("> ")
		_, err := fmt.Scanln(&input)
		if err != nil {
			continue
		}

		parts := strings.Fields(input)
		if len(parts) == 0 {
			continue
		}

		switch parts[0] {
		case "store":
			if len(parts) < 3 {
				fmt.Println("Usage: store <key> <value>")
				continue
			}
			key := parts[1]
			value := strings.Join(parts[2:], " ")

			err := kademlia.Store([]byte(key), []byte(value))
			if err != nil {
				fmt.Printf("Store failed: %v\n", err)
			} else {
				fmt.Printf("Stored: %s = %s\n", key, value)
			}

		case "get":
			if len(parts) < 2 {
				fmt.Println("Usage: get <key>")
				continue
			}
			key := parts[1]

			value, err := kademlia.FindValue([]byte(key))
			if err != nil {
				fmt.Printf("Get failed: %v\n", err)
			} else {
				fmt.Printf("Found: %s = %s\n", key, string(value))
			}

		case "peers":
			count := kademlia.GetPeerCount()
			fmt.Printf("Connected peers: %d\n", count)

		case "quit", "exit":
			fmt.Println("Goodbye!")
			return

		default:
			fmt.Printf("Unknown command: %s\n", parts[0])
		}
	}
}
