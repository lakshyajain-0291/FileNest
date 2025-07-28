package api

import (
	"context"
	"encoding/json"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
)

type PeerAssignmentRequest struct {
	FileEmbed    map[string]float64 `json:"file_embed"`
	FileMetadata map[string]string  `json:"file_metadata"`
	FileLocation string             `json:"file_location"`
	Depth        int                `json:"depth"`
}

type PeerAssignmentResponse struct {
	Results []struct {
		FileEmbed    map[string]float64 `json:"file_embed"`
		FileMetadata map[string]string  `json:"file_metadata"`
		FileLocation string             `json:"file_location"`
	} `json:"results"`
}

const PeerAssignmentProtocol = "/filenest/peer-assignment/1.0.0"

func HandlePeerAssignmentAPI(h host.Host, handler func(PeerAssignmentRequest) (PeerAssignmentResponse, error)) {
	h.SetStreamHandler(PeerAssignmentProtocol, func(s network.Stream) {
		defer s.Close()
		var req PeerAssignmentRequest
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

func SendPeerAssignment(ctx context.Context, h host.Host, peerID string, req PeerAssignmentRequest) (PeerAssignmentResponse, error) {
	var resp PeerAssignmentResponse
	pid, err := peer.Decode(peerID)
	if err != nil {
		return resp, err
	}
	s, err := h.NewStream(ctx, pid, PeerAssignmentProtocol)
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
