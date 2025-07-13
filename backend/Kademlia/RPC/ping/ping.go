package ping

import (
	"context"
	"fmt"
	"log"
	"time"

	libp2p "github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/protocol/ping"
	tcp "github.com/libp2p/go-libp2p/p2p/transport/tcp"
	"github.com/multiformats/go-multiaddr"
)

func Ping(ipv4 string, port int, peerid string) (err error) {

	// TCP only:
	host, err := libp2p.New(libp2p.Transport(tcp.NewTCPTransport))

	if err != nil {
		log.Fatal(err)
		return err
	}
	defer host.Close()

	ipaddr := fmt.Sprintf("/ipv4/%s/tcp/%d/p2p/%s", ipv4, port, peerid)
	targetAddr, err := multiaddr.NewMultiaddr(ipaddr)

	if err != nil {
		log.Fatal(err)
		return err
	}
	defer host.Close()

	addrInfo, err := peer.AddrInfoFromP2pAddr(targetAddr)
	if err != nil {
		log.Fatal(err)
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err = host.Connect(ctx, *addrInfo)
	if err != nil {
		log.Fatal(err)
		return err
	}

	pingService := ping.NewPingService(host)
	result := <-pingService.Ping(ctx, addrInfo.ID)

	if result.Error != nil {
		log.Fatal(err)
		return err
	}

	fmt.Printf("Ping successful! RTT: %v\n", result.RTT)
	return nil

}
