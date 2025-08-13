package helper

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

// Request and Response structures (same as before)
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

// Database models
type Node struct {
	NodeID    int    `gorm:"primaryKey;column:node_id"`
	Embedding []byte `gorm:"column:embedding"`
	CreatedAt time.Time
	UpdatedAt time.Time
}

type Centroid struct {
	ID        uint   `gorm:"primaryKey"`
	NodeID    int    `gorm:"column:node_id;uniqueIndex"`
	Centroid  []byte `gorm:"column:centroid"`
	NodeCount int    `gorm:"column:node_count;default:0"`
	CreatedAt time.Time
	UpdatedAt time.Time
}

type DatabaseService struct {
	db *gorm.DB
}

// Processing service without host
type EmbeddingProcessor struct {
	dbService    *DatabaseService
	currentDepth int
	messageType  string
	hostPeerID   peer.ID // Store the peer ID from main.go
}

// Global variables for tracking
var peerRequestCount = make(map[peer.ID]int)
var peerRequestMutex sync.Mutex
var globalProcessor *EmbeddingProcessor

// Helper functions (same as before)
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

// Database service functions
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

// Processing functions - no network handling, just business logic
func (ep *EmbeddingProcessor) ProcessEmbeddingRequest(request EmbeddingSearchRequest) (*EmbeddingSearchResponse, error) {
	log.Printf("Processing embedding request: type=%s, target=%d, depth=%d",
		request.QueryType, request.TargetNodeID, ep.currentDepth)

	// Check if current node is the target
	myNodeID := byteToInt([]byte(ep.hostPeerID))

	if myNodeID == request.TargetNodeID {
		return ep.handleTargetNodeExecution(request)
	} else {
		// Return forward response
		return &EmbeddingSearchResponse{
			Type:          "forward",
			QueryEmbed:    request.Embed,
			Depth:         request.PrevDepth,
			CurrentNodeID: myNodeID,
			NextNodeID:    request.TargetNodeID,
			HostPeerID:    ep.hostPeerID.String(),
			IsD4:          false,
			Found:         false,
		}, nil
	}
}

func (ep *EmbeddingProcessor) handleTargetNodeExecution(request EmbeddingSearchRequest) (*EmbeddingSearchResponse, error) {
	if ep.currentDepth < 4 {
		// Store embedding and update centroid
		embeddingBytes := embeddingToBytes(request.Embed)
		targetNodeBytes := intToByte(request.TargetNodeID)

		err := ep.dbService.UpsertNodeEmbedding(targetNodeBytes, embeddingBytes)
		if err != nil {
			return nil, fmt.Errorf("failed to store embedding: %w", err)
		}

		err = ep.dbService.updateCentroid(request.TargetNodeID, request.Embed)
		if err != nil {
			return nil, fmt.Errorf("failed to update centroid: %w", err)
		}

		var nextNodeID int
		if request.QueryType == "search" {
			similarPeers, err := ep.dbService.findSimilarPeers(request.Embed, request.Threshold)
			if err != nil {
				return nil, fmt.Errorf("failed to find similar peers: %w", err)
			}

			if len(similarPeers) == 0 {
				newPeerID, err := ep.dbService.assignNewPeer(request.Embed)
				if err != nil {
					return nil, fmt.Errorf("failed to assign new peer: %w", err)
				}
				nextNodeID = newPeerID
			} else {
				nextNodeID = similarPeers[0].NodeID
			}

			log.Printf("Processed search at depth %d, next node: %d", ep.currentDepth, nextNodeID)
		} else {
			log.Printf("Processed store at depth %d", ep.currentDepth)
		}

		return &EmbeddingSearchResponse{
			Type:          "processed",
			QueryEmbed:    request.Embed,
			Depth:         ep.currentDepth,
			CurrentNodeID: request.TargetNodeID,
			NextNodeID:    nextNodeID,
			HostPeerID:    ep.hostPeerID.String(),
			IsD4:          false,
			Found:         true,
		}, nil

	} else {
		// Handle D4 operations
		if request.QueryType == "search" {
			return ep.handleD4Search(request)
		} else {
			return ep.handleD4Store(request)
		}
	}
}

func (ep *EmbeddingProcessor) handleD4Search(request EmbeddingSearchRequest) (*EmbeddingSearchResponse, error) {
	log.Printf("D4 search request for target %d", request.TargetNodeID)

	var allNodes []Node
	result := ep.dbService.db.Find(&allNodes)
	if result.Error != nil || len(allNodes) == 0 {
		return &EmbeddingSearchResponse{
			Type:          "search_response",
			QueryEmbed:    request.Embed,
			Depth:         4,
			CurrentNodeID: request.TargetNodeID,
			NextNodeID:    0,
			HostPeerID:    ep.hostPeerID.String(),
			IsD4:          true,
			Found:         false,
		}, nil
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
		return &EmbeddingSearchResponse{
			Type:          "search_response",
			QueryEmbed:    retrievedEmbed,
			Depth:         4,
			CurrentNodeID: closestNode.NodeID,
			NextNodeID:    0,
			HostPeerID:    ep.hostPeerID.String(),
			IsD4:          true,
			Found:         true,
		}, nil
	}

	return nil, fmt.Errorf("no closest node found")
}

func (ep *EmbeddingProcessor) handleD4Store(request EmbeddingSearchRequest) (*EmbeddingSearchResponse, error) {
	log.Printf("D4 store request for target %d", request.TargetNodeID)

	embeddingBytes := embeddingToBytes(request.Embed)
	targetNodeBytes := intToByte(request.TargetNodeID)

	err := ep.dbService.UpsertNodeEmbedding(targetNodeBytes, embeddingBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to store at D4: %w", err)
	}

	return &EmbeddingSearchResponse{
		Type:          "store_response",
		QueryEmbed:    request.Embed,
		Depth:         4,
		CurrentNodeID: request.TargetNodeID,
		NextNodeID:    0,
		HostPeerID:    ep.hostPeerID.String(),
		IsD4:          true,
		Found:         true,
	}, nil
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

// Initialize processor with host peer ID from main.go
func InitializeProcessor(hostPeerID peer.ID, depth int, msgtype string) (*EmbeddingProcessor, error) {
	dbService, err := NewDatabaseService()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize database: %w", err)
	}

	processor := &EmbeddingProcessor{
		dbService:    dbService,
		currentDepth: depth,
		messageType:  msgtype,
		hostPeerID:   hostPeerID, // Use the peer ID from main.go host
	}

	// Store globally for access
	globalProcessor = processor

	log.Printf("Embedding processor initialized at depth %d, type %s, host peer ID: %s",
		depth, msgtype, hostPeerID.String())

	return processor, nil
}

// Get the host peer ID (for relay layer)
func GetHostPeerID() peer.ID {
	if globalProcessor != nil {
		return globalProcessor.hostPeerID
	}
	return ""
}

// Get the global processor instance
func GetProcessor() *EmbeddingProcessor {
	return globalProcessor
}
