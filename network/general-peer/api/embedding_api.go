package api

import (
	"context"
	"encoding/json"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
)

type EmbeddingRequest struct {
	FileEmbedding map[string]float64 `json:"file_embedding"`
	FileMetadata  map[string]string  `json:"file_metadata"`
	FileLocation  string             `json:"file_location"`
	Depth         int                `json:"depth"`
}

type EmbeddingResponse struct {
	DkTV            string                 `json:"d2tv"`
	PeerInfoContact map[string]interface{} `json:"peerInfoContact,omitempty"`
	AssignedPeer    map[string]interface{} `json:"assignedPeer,omitempty"`
	Depth           int                    `json:"depth"`
}

const EmbeddingProtocol = "/filenest/embedding/1.0.0"

func HandleEmbeddingAPI(h host.Host, handler func(EmbeddingRequest) (EmbeddingResponse, error)) {
	h.SetStreamHandler(EmbeddingProtocol, func(s network.Stream) {
		defer s.Close()
		var req EmbeddingRequest
		if err := json.NewDecoder(s).Decode(&req); err != nil {
			return
		}
		resp, err := handler(req)
		if err != nil {
			return
		}
		json.NewEncoder(s).Encode(resp)
	})
}

func SendEmbedding(ctx context.Context, h host.Host, peerID string, req EmbeddingRequest) (EmbeddingResponse, error) {
	var resp EmbeddingResponse
	pid, err := peer.Decode(peerID)
	if err != nil {
		return resp, err
	}
	s, err := h.NewStream(ctx, pid, EmbeddingProtocol)
	if err != nil {
		return resp, err
	}
	defer s.Close()
	if err := json.NewEncoder(s).Encode(req); err != nil {
		return resp, err
	}
	if err := json.NewDecoder(s).Decode(&resp); err != nil {
		return resp, err
	}
	return resp, nil
}
