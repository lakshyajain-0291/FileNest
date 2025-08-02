package main

import (
	"crypto/sha1"
	findvalue "dht/RPC/find_value"
	"dht/routing_table"
	"encoding/binary"
	"fmt"
	"log"
	"math/big"
	"os"
	"os/signal"
	"syscall"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/p2p/transport/tcp"
)

// write a function to generate a peer id for a device. ensure that once generated the peer id doesn't change for that particular device
var localPeerID int

func hashStringTo160Bits(s string) *big.Int {
	hash := sha1.Sum([]byte(s))

	return new(big.Int).SetBytes(hash[:8])
}

func writeBigIntBinarytoFile(hash *big.Int, filename string) error {
	bytes := hash.Bytes()

	return os.WriteFile(filename, bytes, 0644)
}

func readBigIntBinaryFromFile(filename string) (*big.Int, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	hash := new(big.Int).SetBytes(data)
	return hash, nil
}

func bigIntToBinaryInt(bigNum *big.Int) int {
	// Get raw bytes from big.Int
	bytes := bigNum.Bytes()

	// If we have fewer than 8 bytes, pad with zeros on the left
	if len(bytes) < 8 {
		padded := make([]byte, 8)
		copy(padded[8-len(bytes):], bytes)
		bytes = padded
	}

	// Take only the last 8 bytes (for int64) and convert
	last8Bytes := bytes[len(bytes)-8:]

	// Convert bytes to int64 using big endian
	return int(int64(binary.BigEndian.Uint64(last8Bytes)))
}

func generatePeerID(name string, ipaddr string) int {
	filename := fmt.Sprintf("%s.txt", name)

	// Check if file exists and has content
	if info, err := os.Stat(filename); err == nil && info.Size() > 0 {
		// File exists and has content, read existing peer ID
		fmt.Printf("Reading existing peer ID from %s\n", filename)
		ipHash, err := readBigIntBinaryFromFile(filename)
		if err != nil {
			log.Printf("Error reading existing peer ID: %v", err)
			// Fall through to create new one
		} else {
			peerID := bigIntToBinaryInt(ipHash)
			fmt.Printf("Loaded existing peer ID: %d\n", peerID)
			return peerID
		}
	}

	// File doesn't exist or is empty, create new peer ID
	fmt.Printf("Creating new peer ID for %s\n", ipaddr)
	ipHash := hashStringTo160Bits(ipaddr)

	if err := writeBigIntBinarytoFile(ipHash, filename); err != nil {
		log.Fatalf("Failed to write peer ID to %s: %v", filename, err)
	}

	peerID := bigIntToBinaryInt(ipHash)
	fmt.Printf("Generated and saved new peer ID: %d to %s\n", peerID, filename)
	return peerID
}

// create a global routing table variable which stores and retrives the records in local memory.
var rt *routing_table.RoutingTable

func main() {
	host, err := libp2p.New(libp2p.Transport(tcp.NewTCPTransport),
		libp2p.ListenAddrStrings("/ip4/0.0.0.0/tcp/0"))

	if err != nil {
		log.Fatal(err)
		// return err
	}
	defer host.Close()

	localPeerID = generatePeerID("peerid_file", host.Addrs()[0].String()) //to keep the peer id consistent acoss restart
	rt = routing_table.NewRoutingTable(localPeerID, host)

	// Register a stream handler to intercept messages
	host.SetStreamHandler("/jsonmessages/1.0.0", func(s network.Stream) {
		findvalue.HandleJSONMessages(s, localPeerID, rt) // need to create a global routing table which is stored in memory
	})

	fmt.Printf("Host ID: %d\n", localPeerID)
	fmt.Printf("Listening on: %v\n", host.Addrs())

	// Keep running
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
}
