package ml_client

import (
	"encoding/json"
	"fmt"
	"general-peer/pkg/models"
	"time"

	"github.com/pebbe/zmq4"
)

const (
	MLServiceTimeout = 10 * time.Second
	MaxRetries       = 3
)

// MLClient handles communication with the ML service
type MLClient struct {
	socket *zmq4.Socket
	addr   string
}

// NewMLClient creates a new ML client
func NewMLClient(addr string) (*MLClient, error) {
	// Initialize context if needed (zmq4 may handle this internally)

	// Create REQ (Request) socket - using explicit constant value if needed
	socket, err := zmq4.NewSocket(zmq4.REQ)
	if err != nil {
		return nil, fmt.Errorf("failed to create ZMQ socket: %v", err)
	}

	// Alternative if REQ constant isn't recognized:
	// socket, err := zmq4.NewSocket(3) // 3 is the numeric value for REQ socket type

	// Set socket options
	if err := socket.SetLinger(0); err != nil {
		socket.Close()
		return nil, fmt.Errorf("failed to set linger: %v", err)
	}

	if err := socket.SetRcvtimeo(MLServiceTimeout); err != nil {
		socket.Close()
		return nil, fmt.Errorf("failed to set receive timeout: %v", err)
	}

	// Connect to the ML service
	if err := socket.Connect(addr); err != nil {
		socket.Close()
		return nil, fmt.Errorf("failed to connect to ML service: %v", err)
	}

	return &MLClient{
		socket: socket,
		addr:   addr,
	}, nil
}

// sendRequest handles the ZMQ communication with retries
func (c *MLClient) sendRequest(request map[string]interface{}) (string, error) {
	data, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %v", err)
	}

	var lastErr error
	for i := 0; i < MaxRetries; i++ {
		_, err = c.socket.Send(string(data), 0)
		if err != nil {
			lastErr = fmt.Errorf("failed to send request (attempt %d): %v", i+1, err)
			time.Sleep(time.Second * time.Duration(i+1))
			continue
		}

		response, err := c.socket.Recv(0)
		if err != nil {
			lastErr = fmt.Errorf("failed to receive response (attempt %d): %v", i+1, err)
			time.Sleep(time.Second * time.Duration(i+1))
			continue
		}

		return response, nil
	}

	return "", lastErr
}

// GenerateEmbedding requests embedding generation from ML service
func (c *MLClient) GenerateEmbedding(text string) ([]float64, error) {
	request := map[string]interface{}{
		"type": "embed",
		"text": text,
	}

	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}

	var result struct {
		Embedding []float64 `json:"embedding"`
		Error     string    `json:"error,omitempty"`
	}

	err = json.Unmarshal([]byte(response), &result)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	if result.Error != "" {
		return nil, fmt.Errorf("ML service error: %s", result.Error)
	}

	return result.Embedding, nil
}

// GenerateClusters requests clustering from ML service
func (c *MLClient) GenerateClusters(path string) (*models.ClusterWrapper, error) {
	request := map[string]interface{}{
		"type": "cluster",
		"path": path,
	}

	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}

	var result struct {
		Clusters *models.ClusterWrapper `json:"clusters"`
		Error    string                 `json:"error,omitempty"`
	}

	err = json.Unmarshal([]byte(response), &result)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	if result.Error != "" {
		return nil, fmt.Errorf("ML service error: %s", result.Error)
	}

	return result.Clusters, nil
}

// Close closes the ZMQ socket
func (c *MLClient) Close() error {
	return c.socket.Close()
}
