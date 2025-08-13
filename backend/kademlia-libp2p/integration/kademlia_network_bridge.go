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

type NetworkMessage struct {
    Type      string          `json:"type"`
    Data      json.RawMessage `json:"data"`
    Source    peer.ID         `json:"source"`
    Target    peer.ID         `json:"target,omitempty"`
    Timestamp int64           `json:"timestamp"`
}

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

    // Use the existing Kademlia node's host instead of creating a new one
    hostPeerID := node.GetHost().ID()

    return &KademliaNetworkBridge{
        kademliaNode: node,
        dbService:    dbService,
        hostPeerID:   hostPeerID,
        currentDepth: depth,
        messageType:  msgType,
    }, nil
}

// Rest of the methods remain the same...
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

func (knb *KademliaNetworkBridge) GetHostPeerID() peer.ID {
    return knb.hostPeerID
}

// Add other methods from previous implementation...


func (knb *KademliaNetworkBridge) handleEmbeddingSearch(networkMsg NetworkMessage) error {
    var embedRequest helper.EmbeddingSearchRequest
    if err := json.Unmarshal(networkMsg.Data, &embedRequest); err != nil {
        return fmt.Errorf("failed to unmarshal embedding request: %w", err)
    }

    targetKey := helper.IntToByte(embedRequest.TargetNodeID)

    _, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    contacts, err := knb.kademliaNode.FindNode(targetKey)
    if err != nil {
        return fmt.Errorf("failed to find target node via Kademlia: %w", err)
    }

    for _, contact := range contacts {
        contactNodeID := helper.ByteToInt(contact.NodeID)
        if contactNodeID == embedRequest.TargetNodeID {
            log.Printf("Target node %d found via Kademlia", embedRequest.TargetNodeID)
            return knb.executeEmbeddingLogic(embedRequest)
        }
    }

    return fmt.Errorf("target node %d not reachable", embedRequest.TargetNodeID)
}
// Add this function to access database service externally
func GetDatabaseService() (*helper.DatabaseService, error) {
    return helper.NewDatabaseService()
}

func (knb *KademliaNetworkBridge) handleEmbeddingStore(networkMsg NetworkMessage) error {
    var embedRequest helper.EmbeddingSearchRequest
    if err := json.Unmarshal(networkMsg.Data, &embedRequest); err != nil {
        return fmt.Errorf("failed to unmarshal embedding store request: %w", err)
    }

    embedRequest.QueryType = "store"
    targetKey := helper.IntToByte(embedRequest.TargetNodeID)

    _, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    contacts, err := knb.kademliaNode.FindNode(targetKey)
    if err != nil {
        return fmt.Errorf("failed to find target node via Kademlia: %w", err)
    }

    for _, contact := range contacts {
        contactNodeID := helper.ByteToInt(contact.NodeID)
        if contactNodeID == embedRequest.TargetNodeID {
            log.Printf("Target node %d found via Kademlia for store", embedRequest.TargetNodeID)
            return knb.executeEmbeddingLogic(embedRequest)
        }
    }

    return fmt.Errorf("target node %d not reachable for store", embedRequest.TargetNodeID)
}

func (knb *KademliaNetworkBridge) executeEmbeddingLogic(request helper.EmbeddingSearchRequest) error {
    myNodeID := helper.ByteToInt([]byte(knb.hostPeerID))
    
    if myNodeID == request.TargetNodeID {
        return knb.handleTargetNodeExecution(request)
    } else {
        log.Printf("Forwarding %s message to network layer", request.QueryType)
        return knb.forwardToNetworkLayer(request)
    }
}

func (knb *KademliaNetworkBridge) handleTargetNodeExecution(request helper.EmbeddingSearchRequest) error {
    if knb.currentDepth < 4 {
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

        if request.QueryType == "search" {
            similarPeers, err := knb.dbService.FindSimilarPeers(request.Embed, request.Threshold)
            if err != nil {
                return fmt.Errorf("failed to find similar peers: %w", err)
            }

            var nextNodeID int
            if len(similarPeers) == 0 {
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
        log.Printf("Handling D4 %s operation for target %d", request.QueryType, request.TargetNodeID)
        return nil
    }
}

func (knb *KademliaNetworkBridge) handleKademliaFind(networkMsg NetworkMessage) error {
    var findData struct {
        Key []byte `json:"key"`
    }

    if err := json.Unmarshal(networkMsg.Data, &findData); err != nil {
        return fmt.Errorf("failed to unmarshal find data: %w", err)
    }

    value, err := knb.kademliaNode.FindValue(findData.Key)
    if err != nil {
        return fmt.Errorf("failed to find value: %w", err)
    }

    log.Printf("Found value of length %d for key", len(value))
    return nil
}

func (knb *KademliaNetworkBridge) forwardToNetworkLayer(request helper.EmbeddingSearchRequest) error {
    log.Printf("Forwarding %s request to network layer for target %d", request.QueryType, request.TargetNodeID)
    return nil
}
