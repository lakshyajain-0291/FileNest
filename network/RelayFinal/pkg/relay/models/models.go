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

type EmbeddingSearchRequest struct {
	SourceNodeID []byte    `json:"source_id"`
	SourcePeerID string    `json:"source_peer_id"`
	QueryEmbed   []float64 `json:"embed"`
	Depth        int       `json:"prev_depth"`
	QueryType    string    `json:"query_type"`
	Threshold    float64   `json:"threshold"`
	ResultsCount int       `json:"results_count"`
	TargetNodeID []byte    `json:"target_node_id"`
}

type EmbeddingSearchResponse struct {
	QueryType    string    `json:"type"`
	QueryEmbed   []float64 `json:"query_embed"`
	Depth        int       `json:"depth"`
	SourceNodeID []byte    `json:"source_node_id"`
	SourcePeerID string    `json:"source_peer_id"`
	NextNodeID   []byte    `json:"next_node_id"`
	Found        bool      `json:"found"`
	FileEmbed    []float64 `json:"file_embed"`
}

type PingRequest struct {
	Type string `json:"query_type"`
	Route string `json:"route"`
	ReceiverPeerID string `json:"ReceiverPeerID"`
	SenderNodeID []byte `json:"sender_id"`
	SenderPeerID string `json:"sender_addr"`
	Timestamp    int64  `json:"timestamp"`
}

type PingResponse struct {
	SenderNodeID []byte `json:"sender_id"`
	SenderPeerID string `json:"sender_addr"`
	Timestamp    int64  `json:"timestamp"`
	Success      bool   `json:"success"`
}

//sent to kademlia node
type FindNodeRequest struct {
    SenderNodeID []byte `json:"sender_node_id"`
    SenderPeerID string `json:"sender_peer_id"`
    TargetID     []byte `json:"target_id"`    // The NodeID we want to reach
    Timestamp    int64  `json:"timestamp"`
}

//sent back to user for relaying
type FindNodeResponse struct {
    SenderNodeID []byte     `json:"sender_node_id"`
    SenderPeerID string     `json:"sender_peer_id"`
    ClosestNodes []any `json:"closest_nodes"`  // K closest nodes to TargetID
    Timestamp    int64      `json:"timestamp"`
    Success      bool       `json:"success"`
}

