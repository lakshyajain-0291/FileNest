package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"time"

	libp2p "github.com/libp2p/go-libp2p"
	kaddht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/core/peer"
	tcp "github.com/libp2p/go-libp2p/p2p/transport/tcp"
	"github.com/multiformats/go-multiaddr"
)

// StoreEmbedding stores embedding-derived data in the DHT.
func StoreEmbedding(ipv4 string, port int, peerID string, embedding []byte, value []byte) error {
	host, err := libp2p.New(libp2p.Transport(tcp.NewTCPTransport))
	if err != nil {
		return fmt.Errorf("error creating host: %w", err)
	}
	defer host.Close()

	ctx := context.Background()
	dht, err := kaddht.New(ctx, host)
	if err != nil {
		return fmt.Errorf("error creating DHT: %w", err)
	}

	if err := dht.Bootstrap(ctx); err != nil {
		return fmt.Errorf("error bootstrapping DHT: %w", err)
	}

	addrStr := fmt.Sprintf("/ipv4/%s/tcp/%d/p2p/%s", ipv4, port, peerID)
	targetAddr, err := multiaddr.NewMultiaddr(addrStr)
	if err != nil {
		return fmt.Errorf("invalid multiaddr: %w", err)
	}
	addrInfo, err := peer.AddrInfoFromP2pAddr(targetAddr)
	if err != nil {
		return fmt.Errorf("error AddrInfo: %w", err)
	}

	ctxDial, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := host.Connect(ctxDial, *addrInfo); err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	hash := sha256.Sum256(embedding)
	key := "/filenest/" + hex.EncodeToString(hash[:])

	if err := dht.PutValue(ctx, key, value); err != nil {
		return fmt.Errorf("failed to store value: %w", err)
	}

	log.Printf("Stored key %s with value size %d bytes", key, len(value))
	return nil
}
