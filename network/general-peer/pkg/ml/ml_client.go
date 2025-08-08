package ml

import (
	"encoding/json"
	"fmt"
	"general-peer/pkg/models"

	"github.com/pebbe/zmq4"
	zmq "github.com/pebbe/zmq4"
)

type Client struct {
	socket *zmq.Socket
}

func NewClient(addr string) (*Client, error) {
	socket, err := zmq4.NewSocket(zmq4.Type(zmq4.PULL))
	if err != nil {
		return nil, err
	}
	if err := socket.Connect(addr); err != nil {
		return nil, err
	}
	return &Client{socket: socket}, nil
}

func (c *Client) ReceiveCluster() (*models.ClusterWrapper, error) {
	msg, err := c.socket.Recv(0)
	if err != nil {
		return nil, err
	}
	var wrapper models.ClusterWrapper
	if err := json.Unmarshal([]byte(msg), &wrapper); err != nil {
		return nil, fmt.Errorf("failed to decode ML cluster: %w", err)
	}
	return &wrapper, nil
}

func (c *Client) Close() {
	c.socket.Close()
}
