package helpers

import (
	"encoding/json"
	"fmt"
	"general-peer/pkg/consts"
	"general-peer/pkg/models"
	"log"
	"net"
)

func InitTCPListener(address string) net.Listener {
	listener, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("Failed to start TCP listener: %v", err)
	}
	log.Printf("Listening on TCP %s", address)
	return listener
}

func ListenForMessage(listener net.Listener, msgChan chan models.Message) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			log.Printf("Failed to accept connection: %v", err)
			continue
		}
		go handleConnection(conn, msgChan)
	}
}

func handleConnection(conn net.Conn, msgChan chan models.Message) {
	defer conn.Close()

	buf := make([]byte, consts.MAX_MSG_SIZE)
	n, err := conn.Read(buf)
	if err != nil {
		log.Printf("Read error: %v", err)
		return
	}

	var msg models.Message
	err = json.Unmarshal(buf[:n], &msg)
	if err != nil {
		log.Printf("Unmarshal error: %v", err)
		return
	}

	switch msg.Type {
	case "query":
		if len(msg.QueryEmbed) != consts.EMBED_DIM {
			log.Println("Invalid query vector size")
			return
		}
	case "peer":
		if msg.CurrentPeerID <= 0 {
			log.Println("Invalid peer ID")
			return
		}
		if msg.FileMetadata.Name == "" && msg.Depth == 4 {
			log.Println("Missing file name at depth 4")
			return
		}
	}

	msgChan <- msg
}

func SendTCPMessage(peerAddr string, msg models.Message) error {
	conn, err := net.Dial("tcp", peerAddr)
	if err != nil {
		return fmt.Errorf("dial error: %v", err)
	}
	defer conn.Close()

	msgBytes, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal error: %v", err)
	}

	_, err = conn.Write(msgBytes)
	if err != nil {
		return fmt.Errorf("write error: %v", err)
	}
	log.Printf("Sent %d bytes to %s", len(msgBytes), peerAddr)
	return nil
}

func ForwardQueryTCP(peerId int, queryEmbed []float64, peerAddr string) error {
	if len(queryEmbed) != consts.EMBED_DIM {
		return fmt.Errorf("embedding dimension mismatch: expected %v", consts.EMBED_DIM)
	}

	msg := models.MessageToPeer{
		QueryEmbed: queryEmbed,
		PeerId:     peerId,
	}

	msgBytes, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal error: %v", err)
	}

	conn, err := net.Dial("tcp", peerAddr)
	if err != nil {
		return fmt.Errorf("dial error: %v", err)
	}
	defer conn.Close()

	n, err := conn.Write(msgBytes)
	if err != nil {
		return fmt.Errorf("write error: %v", err)
	}
	log.Printf("Forwarded %d bytes to %s (peerId %d)", n, peerAddr, peerId)
	return nil
}
