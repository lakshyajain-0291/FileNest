package helper

import (
    "bufio"
    "encoding/binary"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "math"
    "os"
    "os/signal"
    "syscall"
    "time"
    "sync"

    "github.com/libp2p/go-libp2p"
    "github.com/libp2p/go-libp2p/core/network"
    "github.com/libp2p/go-libp2p/core/host"
    "github.com/libp2p/go-libp2p/core/peer"
    ws "github.com/libp2p/go-libp2p/p2p/transport/websocket"
    "gorm.io/driver/sqlite"
    "gorm.io/gorm"
)

type EmbeddingSearchRequest struct {
    Source       string    `json:"source"`
    SourceID     int       `json:"source_id"`
    Embed        []float64 `json:"embed"`
    PrevDepth    int       `json:"prev_depth"`
    QueryType    string    `json:"query_type"`
    Threshold    float64   `json:"threshold"`
    ResultsCount int       `json:"results_count"`
    TargetNodeID int       `json:"target_node_id"`
}

type EmbeddingSearchResponse struct {
    Type          string    `json:"type"`
    QueryEmbed    []float64 `json:"query_embed"`
    Depth         int       `json:"depth"`
    CurrentNodeID int       `json:"current_node_id"`
    NextNodeID    int       `json:"next_node_id"`
    HostPeerID    string    `json:"host_peer_id"`
    IsD4          bool      `json:"is_processed"`
    Found         bool      `json:"found"`
}

type Metadata struct {
    Name      string    `json:"name"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

type Node struct {
    NodeID    int       `gorm:"primaryKey;column:node_id"`
    Embedding []byte    `gorm:"column:embedding"`
    CreatedAt time.Time
    UpdatedAt time.Time
}

type Centroid struct {
    ID        uint      `gorm:"primaryKey"`
    NodeID    int       `gorm:"column:node_id;uniqueIndex"`
    Centroid  []byte    `gorm:"column:centroid"`
    NodeCount int       `gorm:"column:node_count;default:0"`
    CreatedAt time.Time
    UpdatedAt time.Time
}

type DatabaseService struct {
    db *gorm.DB
}

var targetNodeID int
var depth int
var currentHost host.Host
var initialHostPeerID peer.ID
var peerRequestCount = make(map[peer.ID]int)
var peerRequestMutex sync.Mutex

func byteToInt(nodeIDByte []byte) int {
    if len(nodeIDByte) == 0 {
        return 0
    }
    if len(nodeIDByte) >= 4 {
        return int(binary.BigEndian.Uint32(nodeIDByte[:4]))
    }
    padded := make([]byte, 4)
    copy(padded[4-len(nodeIDByte):], nodeIDByte)
    return int(binary.BigEndian.Uint32(padded))
}

func intToByte(nodeID int) []byte {
    nodeBytes := make([]byte, 4)
    binary.BigEndian.PutUint32(nodeBytes, uint32(nodeID))
    return nodeBytes
}

func embeddingToBytes(embed []float64) []byte {
    embeddingBytes := make([]byte, len(embed)*8)
    for i, f := range embed {
        bits := math.Float64bits(f)
        binary.BigEndian.PutUint64(embeddingBytes[i*8:(i+1)*8], bits)
    }
    return embeddingBytes
}

func bytesToEmbedding(data []byte) []float64 {
    if len(data)%8 != 0 {
        return nil
    }
    embed := make([]float64, len(data)/8)
    for i := 0; i < len(embed); i++ {
        bits := binary.BigEndian.Uint64(data[i*8 : (i+1)*8])
        embed[i] = math.Float64frombits(bits)
    }
    return embed
}

func cosineSimilarity(a, b []float64) float64 {
    if len(a) != len(b) {
        return -1.0
    }
    var dotProduct, normA, normB float64
    for i := 0; i < len(a); i++ {
        dotProduct += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    if normA == 0 || normB == 0 {
        return 0.0
    }
    return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func NewDatabaseService() (*DatabaseService, error) {
    db, err := gorm.Open(sqlite.Open("kademlia.db"), &gorm.Config{})
    if err != nil {
        return nil, fmt.Errorf("failed to connect to database: %w", err)
    }
    err = db.AutoMigrate(&Node{}, &Centroid{})
    if err != nil {
        return nil, fmt.Errorf("failed to migrate database: %w", err)
    }
    return &DatabaseService{db: db}, nil
}

func (ds *DatabaseService) UpsertNodeEmbedding(nodeIDByte []byte, embedding []byte) error {
    nodeIDInt := byteToInt(nodeIDByte)
    node := Node{
        NodeID:    nodeIDInt,
        Embedding: embedding,
    }
    result := ds.db.Save(&node)
    if result.Error != nil {
        return fmt.Errorf("failed to upsert node: %w", result.Error)
    }
    log.Printf("Upserted node with ID %d", nodeIDInt)
    return nil
}

func (ds *DatabaseService) updateCentroid(currentNodeID int, newEmbedding []float64) error {
    var centroid Centroid
    result := ds.db.Where("node_id = ?", currentNodeID).First(&centroid)
    
    if result.Error != nil {
        if result.Error == gorm.ErrRecordNotFound {
            centroid = Centroid{
                NodeID:    currentNodeID,
                Centroid:  embeddingToBytes(newEmbedding),
                NodeCount: 1,
            }
        } else {
            return fmt.Errorf("failed to get centroid: %w", result.Error)
        }
    } else {
        currentCentroid := bytesToEmbedding(centroid.Centroid)
        if currentCentroid == nil {
            return fmt.Errorf("invalid centroid data")
        }
        newCount := centroid.NodeCount + 1
        updatedCentroid := make([]float64, len(currentCentroid))
        for i := 0; i < len(currentCentroid); i++ {
            updatedCentroid[i] = (currentCentroid[i]*float64(centroid.NodeCount) + newEmbedding[i]) / float64(newCount)
        }
        centroid.Centroid = embeddingToBytes(updatedCentroid)
        centroid.NodeCount = newCount
    }

    result = ds.db.Save(&centroid)
    if result.Error != nil {
        return fmt.Errorf("failed to update centroid: %w", result.Error)
    }
    log.Printf("Updated centroid for node %d (count: %d)", currentNodeID, centroid.NodeCount)
    return nil
}

func (ds *DatabaseService) findSimilarPeers(queryEmbedding []float64, threshold float64) ([]Node, error) {
    var nodes []Node
    result := ds.db.Find(&nodes)
    if result.Error != nil {
        return nil, result.Error
    }
    var similarNodes []Node
    for _, node := range nodes {
        nodeEmbedding := bytesToEmbedding(node.Embedding)
        if nodeEmbedding == nil {
            continue
        }
        similarity := cosineSimilarity(queryEmbedding, nodeEmbedding)
        if similarity >= threshold {
            similarNodes = append(similarNodes, node)
        }
    }
    return similarNodes, nil
}

func (ds *DatabaseService) assignNewPeer(embedding []float64) (int, error) {
    var maxNodeID int
    ds.db.Model(&Node{}).Select("COALESCE(MAX(node_id), 0)").Scan(&maxNodeID)
    newNodeID := maxNodeID + 1
    
    newNode := Node{
        NodeID:    newNodeID,
        Embedding: embeddingToBytes(embedding),
    }
    result := ds.db.Create(&newNode)
    if result.Error != nil {
        return 0, fmt.Errorf("failed to create new peer: %w", result.Error)
    }

    newCentroid := Centroid{
        NodeID:    newNodeID,
        Centroid:  embeddingToBytes(embedding),
        NodeCount: 1,
    }
    result = ds.db.Create(&newCentroid)
    if result.Error != nil {
        log.Printf("Warning: failed to create centroid for new peer: %v", result.Error)
    }
    log.Printf("Assigned new peer with ID %d", newNodeID)
    return newNodeID, nil
}

func isAuthorizedPeer(peerID peer.ID) bool {
    return true
}

func checkRateLimit(peerID peer.ID) bool {
    peerRequestMutex.Lock()
    defer peerRequestMutex.Unlock()
    
    peerRequestCount[peerID]++
    requestCount := peerRequestCount[peerID]
    
    return requestCount <= 100
}

func createResponseWithHostPeerID(responseType string, request EmbeddingSearchRequest, 
    currentNodeID, nextNodeID int, found bool) EmbeddingSearchResponse {
    return EmbeddingSearchResponse{
        Type:          responseType,
        QueryEmbed:    request.Embed,
        Depth:         request.PrevDepth + 1,
        CurrentNodeID: currentNodeID,
        NextNodeID:    nextNodeID,
        HostPeerID:    initialHostPeerID.String(),
        IsD4:          request.PrevDepth >= 3,
        Found:         found,
    }
}

func HandleJSONMessages(dbService *DatabaseService, currentDepth int) network.StreamHandler {
    return func(stream network.Stream) {
        defer stream.Close()

        remotePeer := stream.Conn().RemotePeer()
        rw := bufio.NewReadWriter(bufio.NewReader(stream), bufio.NewWriter(stream))

        if !isAuthorizedPeer(remotePeer) {
            log.Printf("Unauthorized access attempt from peer %s", remotePeer)
            return
        }

        if !checkRateLimit(remotePeer) {
            log.Printf("Rate limit exceeded for peer %s", remotePeer)
            return
        }

        message, err := rw.ReadString('\n')
        if err != nil {
            if err != io.EOF {
                log.Printf("Error reading message from peer %s: %v", remotePeer, err)
            }
            return
        }

        var request EmbeddingSearchRequest
        err = json.Unmarshal([]byte(message), &request)
        if err != nil {
            log.Printf("Error parsing JSON from peer %s: %v", remotePeer, err)
            return
        }

        log.Printf("Received request from peer %s, depth %d, type: %s, target: %d", 
            remotePeer, request.PrevDepth, request.QueryType, request.TargetNodeID)

        myNodeID := byteToInt([]byte(currentHost.ID()))
        
        if myNodeID == request.TargetNodeID {
            handleTargetNodeExecution(dbService, request, rw, currentDepth, remotePeer)
        } else {
            response := createResponseWithHostPeerID("forward", request, myNodeID, request.TargetNodeID, false)
            responseJSON, _ := json.Marshal(response)
            rw.WriteString(string(responseJSON) + "\n")
            rw.Flush()
            log.Printf("Forwarded request to relay layer with host peer ID: %s (from peer: %s)", 
                initialHostPeerID.String(), remotePeer)
        }
    }
}

func handleTargetNodeExecution(dbService *DatabaseService, request EmbeddingSearchRequest, 
    rw *bufio.ReadWriter, currentDepth int, remotePeer peer.ID) {
    
    if currentDepth < 4 {
        log.Printf("Processing embedding from peer %s at depth %d", remotePeer, currentDepth)
        
        embeddingBytes := embeddingToBytes(request.Embed)
        targetNodeBytes := intToByte(request.TargetNodeID)

        err := dbService.UpsertNodeEmbedding(targetNodeBytes, embeddingBytes)
        if err != nil {
            log.Printf("Error storing embedding from peer %s: %v", remotePeer, err)
        }

        err = dbService.updateCentroid(request.TargetNodeID, request.Embed)
        if err != nil {
            log.Printf("Error updating centroid for peer %s: %v", remotePeer, err)
        }

        similarPeers, err := dbService.findSimilarPeers(request.Embed, request.Threshold)
        if err != nil {
            log.Printf("Error finding similar peers for request from %s: %v", remotePeer, err)
        }

        var nextNodeID int
        if len(similarPeers) == 0 {
            newPeerID, err := dbService.assignNewPeer(request.Embed)
            if err != nil {
                log.Printf("Error assigning new peer for request from %s: %v", remotePeer, err)
                nextNodeID = 0
            } else {
                nextNodeID = newPeerID
                log.Printf("Assigned new peer %d for request from %s", newPeerID, remotePeer)
            }
        } else {
            nextNodeID = similarPeers[0].NodeID
            log.Printf("Found similar peer %d for request from %s", nextNodeID, remotePeer)
        }

        response := createResponseWithHostPeerID("processed", request, request.TargetNodeID, nextNodeID, true)
        responseJSON, _ := json.Marshal(response)
        rw.WriteString(string(responseJSON) + "\n")
        rw.Flush()

        log.Printf("Processed at depth %d, next node: %d, host peer ID: %s, from peer: %s", 
            currentDepth, nextNodeID, initialHostPeerID.String(), remotePeer)

    } else {
        if request.QueryType == "search" {
            handleD4Search(dbService, request, rw, remotePeer)
        } else {
            handleD4Store(dbService, request, rw, remotePeer)
        }
    }
}

func handleD4Search(dbService *DatabaseService, request EmbeddingSearchRequest, 
    rw *bufio.ReadWriter, remotePeer peer.ID) {
    
    log.Printf("D4 search request from peer %s for target %d", remotePeer, request.TargetNodeID)
    
    var allNodes []Node
    result := dbService.db.Find(&allNodes)
    if result.Error != nil || len(allNodes) == 0 {
        log.Printf("No nodes found for search request from peer %s", remotePeer)
        
        response := createResponseWithHostPeerID("search_response", request, request.TargetNodeID, 0, false)
        responseJSON, _ := json.Marshal(response)
        rw.WriteString(string(responseJSON) + "\n")
        rw.Flush()
        return
    }

    var closestNode *Node
    maxSimilarity := -1.0

    for i := range allNodes {
        nodeEmbed := bytesToEmbedding(allNodes[i].Embedding)
        if nodeEmbed == nil {
            continue
        }
        similarity := cosineSimilarity(request.Embed, nodeEmbed)
        if similarity > maxSimilarity {
            maxSimilarity = similarity
            closestNode = &allNodes[i]
        }
    }

    if closestNode != nil {
        retrievedEmbed := bytesToEmbedding(closestNode.Embedding)
        response := EmbeddingSearchResponse{
            Type:          "search_response",
            QueryEmbed:    retrievedEmbed,
            Depth:         4,
            CurrentNodeID: closestNode.NodeID,
            NextNodeID:    0,
            HostPeerID:    initialHostPeerID.String(),
            IsD4:          true,
            Found:         true,
        }
        responseJSON, _ := json.Marshal(response)
        rw.WriteString(string(responseJSON) + "\n")
        rw.Flush()
        log.Printf("Found match at D4 for peer %s, closest node: %d (similarity: %.4f)", 
            remotePeer, closestNode.NodeID, maxSimilarity)
    }
}

func handleD4Store(dbService *DatabaseService, request EmbeddingSearchRequest, 
    rw *bufio.ReadWriter, remotePeer peer.ID) {
    
    log.Printf("D4 store request from peer %s for target %d", remotePeer, request.TargetNodeID)
    
    embeddingBytes := embeddingToBytes(request.Embed)
    targetNodeBytes := intToByte(request.TargetNodeID)

    err := dbService.UpsertNodeEmbedding(targetNodeBytes, embeddingBytes)
    if err != nil {
        log.Printf("Error storing at D4 for peer %s: %v", remotePeer, err)
    } else {
        log.Printf("Successfully stored embedding at D4 for peer %s", remotePeer)
    }

    response := createResponseWithHostPeerID("store_response", request, request.TargetNodeID, 0, true)
    responseJSON, _ := json.Marshal(response)
    rw.WriteString(string(responseJSON) + "\n")
    rw.Flush()
}

// Public functions for integration layer
func IntToByte(nodeID int) []byte {
    return intToByte(nodeID)
}

func ByteToInt(nodeIDByte []byte) int {
    return byteToInt(nodeIDByte)
}

func EmbeddingToBytes(embed []float64) []byte {
    return embeddingToBytes(embed)
}

func (ds *DatabaseService) UpdateCentroid(currentNodeID int, newEmbedding []float64) error {
    return ds.updateCentroid(currentNodeID, newEmbedding)
}

func (ds *DatabaseService) FindSimilarPeers(queryEmbedding []float64, threshold float64) ([]Node, error) {
    return ds.findSimilarPeers(queryEmbedding, threshold)
}

func (ds *DatabaseService) AssignNewPeer(embedding []float64) (int, error) {
    return ds.assignNewPeer(embedding)
}

func Execute(depth int, msgtype string) peer.ID {
    dbService, err := NewDatabaseService()
    if err != nil {
        log.Fatal("Failed to initialize database:", err)
    }

    host, err := libp2p.New(
        libp2p.ListenAddrStrings("/ip4/0.0.0.0/tcp/9090/ws"),
        libp2p.Transport(ws.New),
    )
    if err != nil {
        panic(fmt.Sprintf("Failed to create libp2p host: %v", err))
    }
    defer host.Close()

    currentHost = host
    initialHostPeerID = host.ID()

    host.SetStreamHandler("/jsonmessages/1.0.0", HandleJSONMessages(dbService, depth))

    fmt.Printf("Host ID: %s\n", host.ID())
    fmt.Printf("Listening on: %v\n", host.Addrs())
    log.Printf("Node running at depth %d, handling %s operations", depth, msgtype)
    log.Printf("Host Peer ID for relay layer: %s", initialHostPeerID.String())

    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
    <-sigCh

    log.Println("Shutting down...")
    return initialHostPeerID
}

func GetHostPeerID() peer.ID {
    return initialHostPeerID
}
