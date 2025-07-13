package findvalue

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

// FindValue retrieves stored data for an embedding key.
func FindValue(ipv4 string, port int, peerID string, embedding []byte) ([]byte, error) {
	host, err := libp2p.New(libp2p.Transport(tcp.NewTCPTransport))
	if err != nil {
		return nil, fmt.Errorf("error creating host: %w", err)
	}
	defer host.Close()

	ctx := context.Background()
	dht, err := kaddht.New(ctx, host)
	if err != nil {
		return nil, fmt.Errorf("error creating DHT: %w", err)
	}

	if err := dht.Bootstrap(ctx); err != nil {
		return nil, fmt.Errorf("bootstrap failed: %w", err)
	}

	addrStr := fmt.Sprintf("/ipv4/%s/tcp/%d/p2p/%s", ipv4, port, peerID)
	targetAddr, err := multiaddr.NewMultiaddr(addrStr)
	if err != nil {
		return nil, fmt.Errorf("invalid multiaddr: %w", err)
	}
	addrInfo, err := peer.AddrInfoFromP2pAddr(targetAddr)
	if err != nil {
		return nil, fmt.Errorf("AddrInfo error: %w", err)
	}
	ctxDial, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := host.Connect(ctxDial, *addrInfo); err != nil {
		return nil, fmt.Errorf("connect failed: %w", err)
	}

	// Hash the embedding
	hash := sha256.Sum256(embedding)
	key := "/filenest/" + hex.EncodeToString(hash[:])

	val, err := dht.GetValue(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("get value failed: %w", err)
	}
	log.Printf("Retrieved value of length %d bytes for key %s", len(val), key)
	return val, nil
}
