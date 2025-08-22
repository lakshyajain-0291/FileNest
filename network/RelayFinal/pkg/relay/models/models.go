package models

import (
	"encoding/json"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
)

type DepthPeer struct {
	Host      host.Host
	RelayAddr multiaddr.Multiaddr
	RelayID   peer.ID
	Peers     map[peer.ID]string // peer ID to nickname mapping
}

type ReqFormat struct {
	Type      string          `json:"type,omitempty"`
	PeerID    string          `json:"peerid,omitempty"`
	ReqParams json.RawMessage `json:"reqparams,omitempty"`
	Body      json.RawMessage `json:"body,omitempty"`
}

