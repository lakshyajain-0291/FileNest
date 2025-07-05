package helpers

import (
	"encoding/json"
	"errors"
	"general-peer/pkg/models"
	"log"
	"net"
)

// ForwardQuery forwards the message to peers under the specified parent at the given depth.
func ForwardQuery(msg models.Message, depthPeers map[int]map[int][]models.Peer) error {
	if len(msg.QueryEmbed) != 128 {
		return errors.New("embedding dimension mismatch: expected 128")
	}

	parentID := 0 //no parent for depth 1
	if msg.Depth > 1 {
		parentID = msg.CurrentPeerID
	}

	// Looking up depth first and then parent
	depthMap, ok := depthPeers[msg.Depth]
	if !ok {
		return errors.New("no peers configured for this depth")
	}
	peers, ok := depthMap[parentID]
	if !ok || len(peers) == 0 {
		return errors.New("no peers configured under parent for this depth")
	}

	// Send to all peers under this parent:
	for _, peer := range peers { //iterate over all peers at this depth and parent

		peerAddr := &net.UDPAddr{
			IP:   net.ParseIP(peer.IP),
			Port: peer.Port,
		}

		connOut, err := net.DialUDP("udp", nil, peerAddr)
		if err != nil { //error creating the socket
			log.Printf("Error dialing Depth %d peer %v: %v", msg.Depth, peer.ID, err)
			continue
		}
		defer connOut.Close()

		msgBytes, err := json.Marshal(msg)

		_, err = connOut.Write(msgBytes)
		if err != nil {
			log.Printf("Error sending to Depth %d peer %v: %v", msg.Depth, peer.ID, err)
			continue
		}

		log.Printf("Sent message to Depth %d peer %v under parent %d at %v:%v", msg.Depth, peer.ID, parentID, peer.IP, peer.Port)
	}

	return nil
}
