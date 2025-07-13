package findnode

import (
	"context"
	"fmt"
	"log"
	"time"

	libp2p "github.com/libp2p/go-libp2p"
	kaddht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/core/peer"
	tcp "github.com/libp2p/go-libp2p/p2p/transport/tcp"
	"github.com/multiformats/go-multiaddr"
)

// FindNode returns the peers closest to the target ID.
func FindNode(ipv4 string, port int, peerID string, targetPeerID string) error {
	host, err := libp2p.New(libp2p.Transport(tcp.NewTCPTransport))

	if err != nil {
		return fmt.Errorf("host error: %w", err)
	}

	defer host.Close()

	ctx := context.Background()
	dht, err := kaddht.New(ctx, host)
	if err != nil {
		return fmt.Errorf("dht error: %w", err)
	}
	if err := dht.Bootstrap(ctx); err != nil {
		return fmt.Errorf("bootstrap error: %w", err)
	}

	addrStr := fmt.Sprintf("/ipv4/%s/tcp/%d/p2p/%s", ipv4, port, peerID)
	targetAddr, err := multiaddr.NewMultiaddr(addrStr)
	if err != nil {
		return fmt.Errorf("multiaddr error: %w", err)
	}
	addrInfo, err := peer.AddrInfoFromP2pAddr(targetAddr)
	if err != nil {
		return fmt.Errorf("addrinfo error: %w", err)
	}
	ctxDial, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := host.Connect(ctxDial, *addrInfo); err != nil {
		return fmt.Errorf("connect error: %w", err)
	}

	// Parse target peer ID
	targetID, err := peer.Decode(targetPeerID)
	if err != nil {
		return fmt.Errorf("invalid target peer ID: %w", err)
	}

	// Get closest peers
	closestPeers, err := dht.GetClosestPeers(ctx, string(targetID))
	if err != nil {
		return fmt.Errorf("GetClosestPeers failed: %w", err)
	}

	for _, p := range closestPeers {
		log.Printf("Found peer: %s", p)
	}

	return nil
}
