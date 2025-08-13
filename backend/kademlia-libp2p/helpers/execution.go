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
    "context"

    "github.com/libp2p/go-libp2p"
    "github.com/libp2p/go-libp2p/core/network"
    "github.com/libp2p/go-libp2p/core/host"
    "github.com/libp2p/go-libp2p/core/peer"
    ws "github.com/libp2p/go-libp2p/p2p/transport/websocket"
    "gorm.io/driver/sqlite"
    "gorm.io/gorm"
)

// Your existing structs remain the same...
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

// Add Centroid model for tracking peer centroids
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

// Convert byte array to int
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

// Convert int to byte array
func intToByte(nodeID int) []byte {
    nodeBytes := make([]byte, 4)
    binary.BigEndian.PutUint32(nodeBytes, uint32(nodeID))
    return nodeBytes
}

// Convert float64 slice to byte array
func embeddingToBytes(embed []float64) []byte {
    embeddingBytes := make([]byte, len(embed)*8)
    for i, f := range embed {
        bits := math.Float64bits(f)
        binary.BigEndian.PutUint64(embeddingBytes[i*8:(i+1)*8], bits)
    }
    return embeddingBytes
}

// Convert byte array to float64 slice
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

func NewDatabaseService() (*DatabaseService, error) {
    db, err := gorm.Open(sqlite.Open("kademlia.db"), &gorm.Config{})
    if err != nil {
        return nil, fmt.Errorf("failed to connect to database: %w", err)
    }

    // Migrate both Node and Centroid models
    err = db.AutoMigrate(&Node{}, &Centroid{})
    if err != nil {
        return nil, fmt.Errorf("failed to migrate database: %w", err)
    }

    return &DatabaseService{db: db}, nil
}

// Upsert node embedding
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

// Update centroid for the current peer
func (ds *DatabaseService) updateCentroid(currentNodeID int, newEmbedding []float64) error {
    // Get current centroid
    var centroid Centroid
    result := ds.db.Where("node_id = ?", currentNodeID).First(&centroid)
    
    if result.Error != nil {
        if result.Error == gorm.ErrRecordNotFound {
            // Create new centroid
            centroid = Centroid{
                NodeID:    currentNodeID,
                Centroid:  embeddingToBytes(newEmbedding),
                NodeCount: 1,
            }
        } else {
            return fmt.Errorf("failed to get centroid: %w", result.Error)
        }
    } else {
        // Update existing centroid using running average
        currentCentroid := bytesToEmbedding(centroid.Centroid)
        if currentCentroid == nil {
            return fmt.Errorf("invalid centroid data")
        }

        // Calculate new centroid: (old_centroid * count + new_embedding) / (count + 1)
        newCount := centroid.NodeCount + 1
        updatedCentroid := make([]float64, len(currentCentroid))
        
        for i := 0; i < len(currentCentroid); i++ {
            updatedCentroid[i] = (currentCentroid[i]*float64(centroid.NodeCount) + newEmbedding[i]) / float64(newCount)
        }

        centroid.Centroid = embeddingToBytes(updatedCentroid)
        centroid.NodeCount = newCount
    }

    // Save updated centroid
    result = ds.db.Save(&centroid)
    if result.Error != nil {
        return fmt.Errorf("failed to update centroid: %w", result.Error)
    }

    log.Printf("Updated centroid for node %d (count: %d)", currentNodeID, centroid.NodeCount)
    return nil
}

// Find peers above similarity threshold
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

// Assign new peer with centroid as embedding
func (ds *DatabaseService) assignNewPeer(embedding []float64) (int, error) {
    // Generate new node ID (simple incremental approach)
    var maxNodeID int
    ds.db.Model(&Node{}).Select("COALESCE(MAX(node_id), 0)").Scan(&maxNodeID)
    
    newNodeID := maxNodeID + 1
    
    // Create new node with the embedding as its centroid
    newNode := Node{
        NodeID:    newNodeID,
        Embedding: embeddingToBytes(embedding),
    }

    result := ds.db.Create(&newNode)
    if result.Error != nil {
        return 0, fmt.Errorf("failed to create new peer: %w", result.Error)
    }

    // Create corresponding centroid
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

// Send message to parent/gen peer
func sendToGenPeer(parentPeerID peer.ID, response EmbeddingSearchResponse) error {
    if currentHost == nil {
        return fmt.Errorf("host not initialized")
    }

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    stream, err := currentHost.NewStream(ctx, parentPeerID, "/jsonmessages/1.0.0")
    if err != nil {
        return fmt.Errorf("failed to open stream: %w", err)
    }
    defer stream.Close()

    responseJSON, err := json.Marshal(response)
    if err != nil {
        return fmt.Errorf("failed to marshal response: %w", err)
    }

    _, err = stream.Write(append(responseJSON, '\n'))
    if err != nil {
        return fmt.Errorf("failed to write response: %w", err)
    }

    log.Printf("Sent response to parent peer %s", parentPeerID)
    return nil
}

// Calculate cosine similarity between two embedding vectors
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

// Check if current node matches target node ID from Kademlia
func isTargetNode(currentNodeID int, targetNodeID int) bool {
    return currentNodeID == targetNodeID
}

// Handle incoming JSON messages
func HandleJSONMessages(dbService *DatabaseService, currentDepth int) network.StreamHandler {
    return func(stream network.Stream) {
        defer stream.Close()

        RemotePeer := stream.Conn().RemotePeer()
        rw := bufio.NewReadWriter(bufio.NewReader(stream), bufio.NewWriter(stream))

        // Read JSON message
        message, err := rw.ReadString('\n')
        if err != nil {
            if err != io.EOF {
                log.Printf("Error reading message: %v", err)
            }
            return
        }

        // Parse JSON request
        var request EmbeddingSearchRequest
        err = json.Unmarshal([]byte(message), &request)
        if err != nil {
            log.Printf("Error parsing JSON: %v", err)
            return
        }

        log.Printf("Received request from depth %d, type: %s, target: %d", 
            request.PrevDepth, request.QueryType, request.TargetNodeID)

        // Check if this node is the target node from Kademlia routing
        myNodeID := byteToInt([]byte(currentHost.ID()))
        
        if isTargetNode(myNodeID, request.TargetNodeID) {
            // This node is the target - execute the logic
            handleTargetNodeExecution(dbService, request, rw, currentDepth)
        } else {
            // This node is not the target - forward to gen peer
            response := EmbeddingSearchResponse{
                Type:          "forward",
                QueryEmbed:    request.Embed,
                Depth:         request.PrevDepth,
                CurrentNodeID: myNodeID,
                NextNodeID:    request.TargetNodeID,
                IsD4:          false,
                Found:         false,
            }

            responseJSON, _ := json.Marshal(response)
            rw.WriteString(string(responseJSON) + "\n")
            rw.Flush()

            log.Printf("Forwarded request to gen peer - not target node")
        }
    }
}

// Handle execution when this node is the target
func handleTargetNodeExecution(dbService *DatabaseService, request EmbeddingSearchRequest, 
    rw *bufio.ReadWriter, currentDepth int) {
    
    if currentDepth < 4 {
        // Store the embedding
        embeddingBytes := embeddingToBytes(request.Embed)
        targetNodeBytes := intToByte(request.TargetNodeID)

        err := dbService.UpsertNodeEmbedding(targetNodeBytes, embeddingBytes)
        if err != nil {
            log.Printf("Error storing embedding: %v", err)
        }

        // Update centroid of the current peer
        err = dbService.updateCentroid(request.TargetNodeID, request.Embed)
        if err != nil {
            log.Printf("Error updating centroid: %v", err)
        }

        // Check for similar peers above threshold
        similarPeers, err := dbService.findSimilarPeers(request.Embed, request.Threshold)
        if err != nil {
            log.Printf("Error finding similar peers: %v", err)
        }

        var nextNodeID int
        if len(similarPeers) == 0 {
            // No peers above threshold - assign new peer
            newPeerID, err := dbService.assignNewPeer(request.Embed)
            if err != nil {
                log.Printf("Error assigning new peer: %v", err)
                nextNodeID = 0
            } else {
                nextNodeID = newPeerID
            }
        } else {
            // Use most similar peer
            nextNodeID = similarPeers[0].NodeID
        }

        // Send response
        response := EmbeddingSearchResponse{
            Type:          "processed",
            QueryEmbed:    request.Embed,
            Depth:         currentDepth,
            CurrentNodeID: request.TargetNodeID,
            NextNodeID:    nextNodeID,
            IsD4:          false,
            Found:         true,
        }

        responseJSON, _ := json.Marshal(response)
        rw.WriteString(string(responseJSON) + "\n")
        rw.Flush()

        log.Printf("Processed at depth %d, next node: %d", currentDepth, nextNodeID)

    } else {
        // Depth == 4
        if request.QueryType == "search" {
            handleD4Search(dbService, request, rw)
        } else {
            handleD4Store(dbService, request, rw)
        }
    }
}

// Handle search operation at D4
func handleD4Search(dbService *DatabaseService, request EmbeddingSearchRequest, rw *bufio.ReadWriter) {
    log.Printf("Search operation at D4 for target %d", request.TargetNodeID)

    var allNodes []Node
    result := dbService.db.Find(&allNodes)
    if result.Error != nil || len(allNodes) == 0 {
        response := EmbeddingSearchResponse{
            Type:          "search_response",
            QueryEmbed:    request.Embed,
            Depth:         4,
            CurrentNodeID: request.TargetNodeID,
            NextNodeID:    0,
            IsD4:          true,
            Found:         false,
        }

        responseJSON, _ := json.Marshal(response)
        rw.WriteString(string(responseJSON) + "\n")
        rw.Flush()
        return
    }

    // Find closest node using cosine similarity
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

    if closestNode == nil {
        log.Printf("Could not find closest node")
        return
    }

    retrievedEmbed := bytesToEmbedding(closestNode.Embedding)
    
    response := EmbeddingSearchResponse{
        Type:          "search_response",
        QueryEmbed:    retrievedEmbed,
        Depth:         4,
        CurrentNodeID: closestNode.NodeID,
        NextNodeID:    0,
        IsD4:          true,
        Found:         true,
    }

    responseJSON, _ := json.Marshal(response)
    rw.WriteString(string(responseJSON) + "\n")
    rw.Flush()

    log.Printf("Found closest match: node %d (similarity: %.4f)", closestNode.NodeID, maxSimilarity)
}

// Handle store operation at D4
func handleD4Store(dbService *DatabaseService, request EmbeddingSearchRequest, rw *bufio.ReadWriter) {
    log.Printf("Store operation at D4 for target %d", request.TargetNodeID)

    embeddingBytes := embeddingToBytes(request.Embed)
    targetNodeBytes := intToByte(request.TargetNodeID)

    err := dbService.UpsertNodeEmbedding(targetNodeBytes, embeddingBytes)
    if err != nil {
        log.Printf("Error storing at D4: %v", err)
    }

    response := EmbeddingSearchResponse{
        Type:          "store_response",
        QueryEmbed:    request.Embed,
        Depth:         4,
        CurrentNodeID: request.TargetNodeID,
        NextNodeID:    0,
        IsD4:          true,
        Found:         true,
    }

    responseJSON, _ := json.Marshal(response)
    rw.WriteString(string(responseJSON) + "\n")
    rw.Flush()

    log.Printf("Successfully stored embedding at D4")
}

func Execute(depth int, msgtype string) {
    // Initialize database
    dbService, err := NewDatabaseService()
    if err != nil {
        log.Fatal("Failed to initialize database:", err)
    }

    // Create libp2p host
    host, err := libp2p.New(
        libp2p.ListenAddrStrings("/ip4/0.0.0.0/tcp/9090/ws"),
        libp2p.Transport(ws.New),
    )
    if err != nil {
        panic(fmt.Sprintf("Failed to create libp2p host: %v", err))
    }
    defer host.Close()

    // Set global host reference
    currentHost = host

    // Set stream handler with current depth
    host.SetStreamHandler("/jsonmessages/1.0.0", HandleJSONMessages(dbService, depth))

    fmt.Printf("Host ID: %s\n", host.ID())
    fmt.Printf("Listening on: %v\n", host.Addrs())

    log.Printf("Node running at depth %d, handling %s operations", depth, msgtype)

    // Keep running
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
    <-sigCh

    log.Println("Shutting down...")
}
