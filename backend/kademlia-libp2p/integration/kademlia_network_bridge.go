package integration

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"

    "github.com/libp2p/go-libp2p/core/peer"
    "kademlia-libp2p/kademlia"
    "kademlia-libp2p/helpers"
)

// Message types from network layer
type NetworkMessage struct {
    Type      string          `json:"type"`
    Data      json.RawMessage `json:"data"`
    Source    peer.ID         `json:"source"`
    Target    peer.ID         `json:"target,omitempty"`
    Timestamp int64           `json:"timestamp"`
}

// KademliaNetworkBridge handles integration between network layer and Kademlia
type KademliaNetworkBridge struct {
    kademliaNode   *kademlia.Node
    dbService      *helper.DatabaseService
    hostPeerID     peer.ID
    currentDepth   int
    messageType    string
}

func NewKademliaNetworkBridge(node *kademlia.Node, depth int, msgType string) (*KademliaNetworkBridge, error) {
    // Initialize database service for embedding operations
    dbService, err := helper.NewDatabaseService()
    if err != nil {
        return nil, fmt.Errorf("failed to initialize database service: %w", err)
    }

    // Get host peer ID from the execute function
    hostPeerID := helper.Execute(depth, msgType)

    return &KademliaNetworkBridge{
        kademliaNode: node,
        dbService:    dbService,
        hostPeerID:   hostPeerID,
        currentDepth: depth,
        messageType:  msgType,
    }, nil
}

// ProcessNetworkMessage handles incoming messages from network layer
func (knb *KademliaNetworkBridge) ProcessNetworkMessage(networkMsg NetworkMessage) error {
    log.Printf("Processing network message of type: %s from peer: %s", networkMsg.Type, networkMsg.Source)

    switch networkMsg.Type {
    case "embedding_search":
        return knb.handleEmbeddingSearch(networkMsg)
    case "embedding_store":
        return knb.handleEmbeddingStore(networkMsg)
    case "kademlia_find":
        return knb.handleKademliaFind(networkMsg)
    default:
        return fmt.Errorf("unknown message type: %s", networkMsg.Type)
    }
}

// Handle embedding search requests
func (knb *KademliaNetworkBridge) handleEmbeddingSearch(networkMsg NetworkMessage) error {
    var embedRequest helper.EmbeddingSearchRequest
    if err := json.Unmarshal(networkMsg.Data, &embedRequest); err != nil {
        return fmt.Errorf("failed to unmarshal embedding request: %w", err)
    }

    // Convert target node ID from embedding request to Kademlia key
    targetKey := helper.IntToByte(embedRequest.TargetNodeID)

    // Use Kademlia to find the target node
    _, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    contacts, err := knb.kademliaNode.FindNode(targetKey)
    if err != nil {
        return fmt.Errorf("failed to find target node via Kademlia: %w", err)
    }

    // Check if any of the found contacts match our target
    targetFound := false
    for _, contact := range contacts {
        contactNodeID := helper.ByteToInt(contact.NodeID)
        if contactNodeID == embedRequest.TargetNodeID {
            targetFound = true
            log.Printf("Target node %d found via Kademlia", embedRequest.TargetNodeID)

            // Execute embedding logic since we found the target
            return knb.executeEmbeddingLogic(embedRequest)
        }
    }

    if !targetFound {
        log.Printf("Target node %d not found in Kademlia network", embedRequest.TargetNodeID)
        return fmt.Errorf("target node %d not reachable", embedRequest.TargetNodeID)
    }

    return nil
}

// Handle embedding store requests (uses modified store RPC in execution.go)
func (knb *KademliaNetworkBridge) handleEmbeddingStore(networkMsg NetworkMessage) error {
    var embedRequest helper.EmbeddingSearchRequest
    if err := json.Unmarshal(networkMsg.Data, &embedRequest); err != nil {
        return fmt.Errorf("failed to unmarshal embedding store request: %w", err)
    }

    // Set query type to store
    embedRequest.QueryType = "store"

    // Convert target node ID from embedding request to Kademlia key
    targetKey := helper.IntToByte(embedRequest.TargetNodeID)

    // Use Kademlia to find the target node
    _, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    contacts, err := knb.kademliaNode.FindNode(targetKey)
    if err != nil {
        return fmt.Errorf("failed to find target node via Kademlia: %w", err)
    }

    // Check if any of the found contacts match our target
    targetFound := false
    for _, contact := range contacts {
        contactNodeID := helper.ByteToInt(contact.NodeID)
        if contactNodeID == embedRequest.TargetNodeID {
            targetFound = true
            log.Printf("Target node %d found via Kademlia for store operation", embedRequest.TargetNodeID)

            // Execute embedding store logic since we found the target
            return knb.executeEmbeddingLogic(embedRequest)
        }
    }

    if !targetFound {
        log.Printf("Target node %d not found for store operation", embedRequest.TargetNodeID)
        return fmt.Errorf("target node %d not reachable for store", embedRequest.TargetNodeID)
    }

    return nil
}

// Execute embedding logic when target node is found
func (knb *KademliaNetworkBridge) executeEmbeddingLogic(request helper.EmbeddingSearchRequest) error {
    log.Printf("Executing embedding logic for target node %d at depth %d, type: %s", 
        request.TargetNodeID, knb.currentDepth, request.QueryType)

    // Check if current node is the target
    myNodeID := helper.ByteToInt([]byte(knb.hostPeerID))
    
    if myNodeID == request.TargetNodeID {
        // This node is the target - execute the embedding logic
        // The execution.go handles both search and store operations internally
        return knb.handleTargetNodeExecution(request)
    } else {
        // Forward the message through the network layer
        log.Printf("Forwarding %s message to network layer - not target node", request.QueryType)
        return knb.forwardToNetworkLayer(request)
    }
}

// Handle execution when this node is the target
func (knb *KademliaNetworkBridge) handleTargetNodeExecution(request helper.EmbeddingSearchRequest) error {
    if knb.currentDepth < 4 {
        // Store embedding and update centroid (for both search and store operations)
        embeddingBytes := helper.EmbeddingToBytes(request.Embed)
        targetNodeBytes := helper.IntToByte(request.TargetNodeID)

        err := knb.dbService.UpsertNodeEmbedding(targetNodeBytes, embeddingBytes)
        if err != nil {
            return fmt.Errorf("failed to store embedding: %w", err)
        }

        err = knb.dbService.UpdateCentroid(request.TargetNodeID, request.Embed)
        if err != nil {
            return fmt.Errorf("failed to update centroid: %w", err)
        }

        // Only find similar peers and assign new peers for search operations
        if request.QueryType == "search" {
            similarPeers, err := knb.dbService.FindSimilarPeers(request.Embed, request.Threshold)
            if err != nil {
                return fmt.Errorf("failed to find similar peers: %w", err)
            }

            var nextNodeID int
            if len(similarPeers) == 0 {
                // Assign new peer
                newPeerID, err := knb.dbService.AssignNewPeer(request.Embed)
                if err != nil {
                    return fmt.Errorf("failed to assign new peer: %w", err)
                }
                nextNodeID = newPeerID
            } else {
                nextNodeID = similarPeers[0].NodeID
            }

            log.Printf("Processed search at depth %d, next node: %d", knb.currentDepth, nextNodeID)
        } else {
            log.Printf("Processed store at depth %d", knb.currentDepth)
        }

        return nil

    } else {
        // Handle D4 operations - execution.go handles both search and store
        log.Printf("Handling D4 %s operation for target %d", request.QueryType, request.TargetNodeID)
        
        // The modified store RPC in execution.go will handle the actual storage/retrieval
        // No need for separate Kademlia store operations
        return nil
    }
}

// Handle Kademlia find operations (only for non-embedding data)
func (knb *KademliaNetworkBridge) handleKademliaFind(networkMsg NetworkMessage) error {
    var findData struct {
        Key []byte `json:"key"`
    }

    if err := json.Unmarshal(networkMsg.Data, &findData); err != nil {
        return fmt.Errorf("failed to unmarshal find data: %w", err)
    }

    // Use Kademlia to find the value
    value, err := knb.kademliaNode.FindValue(findData.Key)
    if err != nil {
        return fmt.Errorf("failed to find value: %w", err)
    }

    log.Printf("Found value of length %d for key", len(value))
    return nil
}

// Forward message to network layer
func (knb *KademliaNetworkBridge) forwardToNetworkLayer(request helper.EmbeddingSearchRequest) error {
    // This should call your network layer's send function
    log.Printf("Forwarding %s request to network layer for target %d", request.QueryType, request.TargetNodeID)
    return nil
}
