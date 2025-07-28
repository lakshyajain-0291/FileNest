package api

import (
	"context"
	"encoding/json"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
)

type QueryRequest struct {
	Query string `json:"query"`
}

type ClusterResponse struct {
	Cluster struct {
		Centroid  map[string]float64 `json:"centroid"`
		Filenames []string           `json:"filenames"`
		NFiles    int                `json:"nfiles"`
		NMissing  int                `json:"nmissing"`
	} `json:"cluster"`
}

const QueryProtocol = "/filenest/query/1.0.0"

func HandleQueryAPI(h host.Host, handler func(QueryRequest) (ClusterResponse, error)) {
	h.SetStreamHandler(QueryProtocol, func(s network.Stream) {
		defer s.Close()
		var req QueryRequest
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

func SendQuery(ctx context.Context, h host.Host, peerID string, req QueryRequest) (ClusterResponse, error) {
	var resp ClusterResponse
	pid, err := peer.Decode(peerID)
	if err != nil {
		return resp, err
	}
	s, err := h.NewStream(ctx, pid, QueryProtocol)
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
