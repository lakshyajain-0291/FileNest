package helpers

import (
	"context"
	"encoding/json"
	"fmt"
	"general-peer/pkg/models"
	"log"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
	webrtc "github.com/libp2p/go-libp2p/p2p/transport/webrtc"
	ma "github.com/multiformats/go-multiaddr"
)

func CreateWebRTCHost(port int) (host.Host, error) {
    priv, _, err := crypto.GenerateKeyPair(crypto.Ed25519, 0)
    if err != nil {
        return nil, err
    }

    // Use webrtc-direct instead of webrtc
    addr, err := ma.NewMultiaddr(fmt.Sprintf("/ip4/0.0.0.0/udp/%d/webrtc-direct", port))
    if err != nil {
        return nil, err
    }

    // Pass the constructor, not the instance!
    return libp2p.New(
        libp2p.Identity(priv),
        libp2p.ListenAddrs(addr),
        libp2p.Transport(webrtc.New),
    )
}

func Decoder(s network.Stream, msgChan chan models.Message){
    var msg models.Message
    if err := json.NewDecoder(s).Decode(&msg); err != nil {
        log.Printf("WebRTC decode error (host): %v", err)
        s.Reset()
        return
    }
    s.Close()
    msgChan <- msg
}

func GetHostAddress(h host.Host) ma.Multiaddr {
    addrs := h.Addrs()
    if len(addrs) == 0 {
        return nil
    }

    return addrs[0].Encapsulate(ma.StringCast("/p2p/" + h.ID().String()))
}

func SendWebRTCMessage(h host.Host, target ma.Multiaddr, msg models.Message, endpoint protocol.ID) error {
    peerInfo, err := peer.AddrInfoFromP2pAddr(target)
    if err != nil {
        return err
    }


    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    if err := h.Connect(ctx, *peerInfo); err != nil {
        return err
    }

    s, err := h.NewStream(ctx, peerInfo.ID, endpoint)
    if err != nil {
        return err
    }
    defer s.Close()

    return json.NewEncoder(s).Encode(msg)
}
