package peerhelpers

import (
	"encoding/json"
	"fmt"
	"log"
	"final/network/RelayFinal/pkg/generalpeer/models"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
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
