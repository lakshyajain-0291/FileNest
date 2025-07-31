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
	file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	peerid, err := readBigIntBinaryFromFile(filename)
	if err != nil {
		fmt.Errorf("Could not retrieve the peer id: %w", err)
	}
	if peerid == nil {
		ipHash := hashStringTo160Bits(ipaddr)
		writeBigIntBinarytoFile(ipHash, filename)
		return int(ipHash.Int64())
	}
	ipHash, err := readBigIntBinaryFromFile(filename)
	if err != nil {
		fmt.Errorf("Could not retrieve the peer id: %w", err)
	}
    peerID := bigIntToBinaryInt(ipHash)

	return peerID
}

// create a global routing table variable which stores and retrives the records in local memory.
var rt *routing_table.RoutingTable

func main() {
	host, err := libp2p.New(libp2p.Transport(tcp.NewTCPTransport))

	if err != nil {
		log.Fatal(err)
		// return err
	}
	defer host.Close()

    localPeerID = generatePeerID("peerid_file", host.Addrs()[0].String())

	// Register a stream handler to intercept messages
	host.SetStreamHandler("/jsonmessages/1.0.0", func(s network.Stream) {
		findvalue.HandleJSONMessages(s, localPeerID, rt) // need to create a global routing table which is stored in memory
	})

	fmt.Printf("Host ID: %s\n", localPeerID)
	fmt.Printf("Listening on: %v\n", host.Addrs())

	// Keep running
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
}
