package main

import (
    "bufio"
    "context"
    "flag"
    "fmt"
    "os"
    "strings"
    "time"

    "github.com/libp2p/go-libp2p"
    "github.com/libp2p/go-libp2p/core/peer"
    "github.com/multiformats/go-multiaddr"

    "kademlia-libp2p/kademlia"
)

func main() {
    port := flag.Int("port", 0, "Port to listen on")
    bootstrap := flag.String("bootstrap", "", "Bootstrap peer addresses")
    dataDir := flag.String("datadir", "./data", "Data directory")
    flag.Parse()

    // Create libp2p host
    host, err := libp2p.New(
        libp2p.ListenAddrStrings(fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", *port)),
    )
    if err != nil {
        panic(err)
    }
    defer host.Close()

    fmt.Printf("Peer ID: %s\n", host.ID())
    for _, addr := range host.Addrs() {
        fmt.Printf("Address: %s/p2p/%s\n", addr, host.ID())
    }

    // Create Kademlia node
    node, err := kademlia.NewNode(host, *dataDir)
    if err != nil {
        panic(err)
    }
    defer node.Stop()

    // Bootstrap
    if *bootstrap != "" {
        var bootstrapPeers []peer.ID
        for _, addrStr := range strings.Split(*bootstrap, ",") {
            maddr, err := multiaddr.NewMultiaddr(strings.TrimSpace(addrStr))
            if err != nil {
                continue
            }
            peerInfo, err := peer.AddrInfoFromP2pAddr(maddr)
            if err != nil {
                continue
            }
            bootstrapPeers = append(bootstrapPeers, peerInfo.ID)
            
            ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
            host.Connect(ctx, *peerInfo)
            cancel()
        }
        
        if len(bootstrapPeers) > 0 {
            node.Bootstrap(bootstrapPeers)
        }
    }

    // CLI
    fmt.Println("\nCommands: store <key> <value>, get <key>, peers, ping <peer_id>, quit")
    scanner := bufio.NewScanner(os.Stdin)
    
    for {
        fmt.Print("> ")
        if !scanner.Scan() {
            break
        }
        
        parts := strings.Fields(scanner.Text())
        if len(parts) == 0 {
            continue
        }

        switch parts[0] {
        case "store":
            if len(parts) >= 3 {
                key := parts[1]
                value := strings.Join(parts[2:], " ")
                err := node.Store([]byte(key), []byte(value))
                if err != nil {
                    fmt.Printf("Error: %v\n", err)
                } else {
                    fmt.Printf("Stored: %s\n", key)
                }
            }
        case "get":
            if len(parts) >= 2 {
                key := parts[1]
                value, err := node.FindValue([]byte(key))
                if err != nil {
                    fmt.Printf("Error: %v\n", err)
                } else {
                    fmt.Printf("Found: %s = %s\n", key, string(value))
                }
            }
        case "peers":
            fmt.Printf("Connected peers: %d\n", node.GetPeerCount())
        case "ping":
            if len(parts) >= 2 {
                peerID, err := peer.Decode(parts[1])
                if err != nil {
                    fmt.Printf("Invalid peer ID: %v\n", err)
                    continue
                }
                err = node.Ping(peerID)
                if err != nil {
                    fmt.Printf("Ping failed: %v\n", err)
                } else {
                    fmt.Printf("Ping successful\n")
                }
            }
        case "quit":
            return
        }
    }
}
