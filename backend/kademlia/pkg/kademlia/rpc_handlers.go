package kademlia

import (
	"fmt"
	"kademlia/pkg/types"
	"time"
)

// PING handlers (same as before)
func (k *KademliaNode) HandlePing(req *types.PingRequest) (*types.PingResponse, error) {
    if req == nil {
        return nil, fmt.Errorf("ping request cannot be nil")
    }
    
    // Update routing table with sender information
    sender := types.PeerInfo{
        NodeID:      req.SenderNodeID,
        PeerID: req.SenderPeerID,
    }
    k.routingTable.Update(sender)
    
    // Create successful response
    response := &types.PingResponse{
        SenderNodeID:   k.NodeID,
        SenderPeerID: k.PeerID,
        Timestamp:  time.Now().UnixNano(),
        Success:    true,
    }
    
    return response, nil
}

func (k *KademliaNode) Ping(targetNodeID []byte) (*types.PingResponse, error) {
    if targetNodeID == nil {
        return nil, fmt.Errorf("target node ID cannot be nil")
    }
    
    req := &types.PingRequest{
        SenderNodeID:   k.NodeID,
        SenderPeerID: k.PeerID,
        Timestamp:  time.Now().UnixNano(),
    }
    
    response, err := k.network.SendPing(targetNodeID, req)
    if err != nil {
        return nil, fmt.Errorf("ping failed: %w", err)
    }
    
    if response.Success {
        target := types.PeerInfo{
            NodeID:      response.SenderNodeID,
            PeerID: response.SenderPeerID,
        }
        k.routingTable.Update(target)
    }
    
    return response, nil
}

// EMBEDDING SEARCH handlers - using only your structs
func (k *KademliaNode) HandleEmbeddingSearch(req *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
    if req == nil {
        return nil, fmt.Errorf("embedding search request cannot be nil")
    }
    
    // Update routing table with source node
    if len(req.SourceNodeID) > 0 {
        sourceNode := types.PeerInfo{
            NodeID:      req.SourceNodeID,
            PeerID: req.SourcePeerID,
        }
        k.routingTable.Update(sourceNode)
    }
    
    // Search for similar embeddings in local storage. there will be a sqlite database which will be used to store embeddings to node id mapping
    /*
	            WORK NEEDED HERE
	*/
	similarEmbeddings, err := k.storage.FindSimilar(req.QueryEmbed, req.Threshold, req.ResultsCount)
    if err != nil {
        return nil, fmt.Errorf("local search failed: %w", err)
    }
    
    response := &types.EmbeddingSearchResponse{
        QueryType:    req.QueryType,
        QueryEmbed:   req.QueryEmbed,
        Depth:        req.Depth + 1,
        SourceNodeID: k.NodeID,
        SourcePeerID: k.PeerID,
        Found:        len(similarEmbeddings) > 0,
    }
    
    // If we found similar embeddings, return the best match
    if len(similarEmbeddings) > 0 {
        response.FileEmbed = similarEmbeddings[0].Embedding
        return response, nil
    }
    
    // If not found locally, find next best node to forward to
    nextNode := k.findNextNodeForSearch(req.QueryEmbed, req.SourceNodeID)
    if nextNode != nil {
        response.NextNodeID = nextNode.NodeID
    }
    
    return response, nil
}

func (k *KademliaNode) EmbeddingSearch(req *types.EmbeddingSearchRequest) (*types.EmbeddingSearchResponse, error) {
    if req == nil {
        return nil, fmt.Errorf("embedding search request cannot be nil")
    }
    
    // Set source information if not already set
    if len(req.SourceNodeID) == 0 {
        req.SourceNodeID = k.NodeID
        req.SourcePeerID = k.PeerID
    }
    
    response, err := k.network.SendEmbeddingSearch(req.TargetNodeID, req)
    if err != nil {
        return nil, fmt.Errorf("embedding search failed: %w", err)
    }
    
    // Update routing table with response source
    if len(response.SourceNodeID) > 0 {
        responseNode := types.PeerInfo{
            NodeID:      response.SourceNodeID,
            PeerID: response.SourcePeerID,
        }
        k.routingTable.Update(responseNode)
    }
    
    return response, nil
}

// Helper functions
func (k *KademliaNode) findNextNodeForSearch(queryEmbed []float64, excludeNodeID []byte) *types.PeerInfo {
    allNodes := k.routingTable.GetNodes()
    
    for _, node := range allNodes {
        // Skip the source node
        if string(node.NodeID) == string(excludeNodeID) {
            continue
        }
        
        // Return first available node (you can enhance this logic)
        return &node
    }
    
    return nil
}

