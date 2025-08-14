package kademlia

import "kademlia/pkg/types"

// Main Kademlia interface that network layer will use
type Kademlia interface {
    // Handle incoming requests from network layer
    HandlePing(req *types.PingRequest) (*types.PingResponse, error)
    
    // Initiate outbound requests (network layer handles actual sending)
    Ping(targetNodeID []byte) (*types.PingResponse, error)
}

// Network callback interface - how Kademlia talks back to network layer
type NetworkInterface interface {
    SendPing(targetNodeID []byte, req *types.PingRequest) (*types.PingResponse, error)
}