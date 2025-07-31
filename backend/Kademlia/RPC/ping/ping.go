package ping

// import (
// 	"context"
// 	"fmt"

// 	"github.com/libp2p/go-libp2p/core/host"
// 	"github.com/libp2p/go-libp2p/core/peer"
// 	"github.com/libp2p/go-libp2p/p2p/protocol/ping"
// 	"github.com/multiformats/go-multiaddr"
// )

// func Ping(ctx context.Context, host host.Host, contact *routing_table.Contact) (ping.Result, error) {
// 	fullAddr := multiaddr.Join(contact.Address, multiaddr.StringCast("/p2p/"+contact.ID.String()))
// 	addrInfo, err := peer.AddrInfoFromP2pAddr(fullAddr)
// 	if err != nil {
// 		return ping.Result{}, fmt.Errorf("invalid addrinfo: %w", err)
// 	}

// 	if err := host.Connect(ctx, *addrInfo); err != nil {
// 		return ping.Result{}, fmt.Errorf("connect failed: %w", err)
// 	}

// 	pingService := ping.NewPingService(host)
// 	result := <-pingService.Ping(ctx, addrInfo.ID)

// 	if result.Error != nil {
// 		return result, result.Error
// 	}

// 	return result, nil
// }
