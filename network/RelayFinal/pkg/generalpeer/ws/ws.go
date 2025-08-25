package ws

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"network/pkg/generalpeer/models"
	"sync"

	"github.com/gorilla/websocket"
)

// WebSocketTransport manages WS send/receive
type WebSocketTransport struct {
	addr      string
	upgrader  websocket.Upgrader
	clients   map[*websocket.Conn]bool
	mu        sync.Mutex
	closeChan chan struct{}
}

// NewWebSocketTransport creates a WS transport bound to addr
func NewWebSocketTransport(addr string) *WebSocketTransport {
	return &WebSocketTransport{
		addr: addr,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true }, // allow all for now
		},
		clients:   make(map[*websocket.Conn]bool),
		closeChan: make(chan struct{}),
	}
}

// StartReceiver starts a WS server and pushes messages into msgChan
func (w *WebSocketTransport) StartPeerReceiver(msgChan chan models.Message) error {
	http.HandleFunc("/peer", func(rw http.ResponseWriter, r *http.Request) {
		conn, err := w.upgrader.Upgrade(rw, r, nil)
		if err != nil {
			log.Println("[NET] WS upgrade error:", err)
			return
		}

		w.mu.Lock()
		w.clients[conn] = true
		w.mu.Unlock()

		log.Println("[NET] New WebSocket client connected")

		go func(c *websocket.Conn) {
			defer func() {
				w.mu.Lock()
				delete(w.clients, c)
				w.mu.Unlock()
				c.Close()
			}()

			for {
				select {
				case <-w.closeChan:
					return
				default:
					_, data, err := c.ReadMessage()
					if (err != nil) {
						log.Println("[NET] Peer WS read error:", err)
						return
					}
					var msg models.Message
					if err := json.Unmarshal(data, &msg); err != nil {
						log.Println("[NET] JSON decode error:", err)
						continue
					}
					msgChan <- msg
				}
			}
		}(conn)
	})

	log.Println("[NET] WS listening on", w.addr)
	return http.ListenAndServe(w.addr, nil)
}

func (w *WebSocketTransport) StartMLReceiver(mlChan chan models.ClusterWrapper) error {
	http.HandleFunc("/ml", func(rw http.ResponseWriter, r *http.Request) {
		conn, err := w.upgrader.Upgrade(rw, r, nil) // changes to websockets
		if err != nil {
			log.Printf("[NET] WS upgrade error: %v", err.Error())
			return
		}

		w.mu.Lock()
		w.clients[conn] = true
		w.mu.Unlock()

		log.Println("[NET] New WebSocket client connected")

		go func(c *websocket.Conn) {
			defer func() {
				w.mu.Lock()
				delete(w.clients, c)
				w.mu.Unlock()
				c.Close()
			}()

			for {
				select {
				case <-w.closeChan:
					return

				default:
					_, data, err := c.ReadMessage() // blocks here until msg sent
					if err != nil {
						log.Printf("[NET] ML WS read error: %v", err.Error())
						return // goes back to top of select
					}

					var msg models.ClusterWrapper
					if err := json.Unmarshal(data, &msg); err != nil {
						log.Printf("[NET] JSON decode error: %v", err.Error())
						continue
					}
					mlChan <- msg
				}
			}
		}(conn)
	})
	log.Println("[NET] WS listening on", w.addr)
	return http.ListenAndServe(w.addr, nil)
}

// SendMessage connects to another WS server and sends a message
func (w *WebSocketTransport) SendMessage(dest string, msg models.Message) error {
	c, _, err := websocket.DefaultDialer.Dial(dest, nil)
	if err != nil {
		return fmt.Errorf("[NET] dial error: %w", err)
	}
	defer c.Close()

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("[NET] marshal error: %w", err)
	}

	return c.WriteMessage(websocket.TextMessage, data)
}

// Close stops the transport
func (w *WebSocketTransport) Close() error {
	close(w.closeChan)
	w.mu.Lock()
	for conn := range w.clients {
		conn.Close()
	}
	w.mu.Unlock()
	return nil
}