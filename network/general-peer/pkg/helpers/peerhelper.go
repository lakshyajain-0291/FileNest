package helpers

import (
	"encoding/json"
	"errors"
	"fmt"
	"general-peer/pkg/consts"
	"general-peer/pkg/models"
	"log"
	"net"
)



func InitPeer(peer *models.Peer) (*net.UDPConn, *net.UDPAddr){
	peerAddr := &net.UDPAddr{IP: net.ParseIP(peer.IP), Port: peer.Port}

	conn, err := net.ListenUDP("udp", peerAddr)
	if(err != nil){
		log.Printf("Failed to initialize peer with given params with err: %v", err.Error())
		return nil, nil
	}
	log.Printf("Peer successfully initialized with given params: %v", peerAddr)
	return conn, peerAddr
}

func ListenForMessage(conn *net.UDPConn, msgChannel *chan models.Message) error{
	msgb := make([]byte, consts.MAX_MSG_SIZE)
	msg := &models.Message{}

	for {
		n,incomingAddr, err := conn.ReadFromUDP(msgb)
		if(err != nil){
			log.Printf("Error during reading of message: %v\n", err.Error())
		}

		err = json.Unmarshal(msgb[:n], msg)
		if(err != nil){
			log.Printf("Error during unmarshalling of read data: %v\n", err.Error())
		}
		log.Printf("Recieved a message of bytes: %v from address: %v\n", n, incomingAddr)

	switch (msg.Type) {
	case "query": 
		if len(msg.QueryEmbed) != consts.EMBED_DIM { 
			return errors.New("query embedding dimensions do not match")
		}

	case "peer": 
		if msg.CurrentPeerID <= 0 { // checks if the peer ID is valid or not
			return errors.New("invalid peer ID")
		}

		if (msg.FileMetadata.Name == "" && msg.Depth == 4) { //checks for empty files
			return errors.New("file name is required")
		}
	}
	log.Printf("going to put msg into msgChannel")
	*msgChannel <- *msg

	msgb = make([]byte, consts.MAX_MSG_SIZE) // resets msgb to take in further values
	msg = &models.Message{} // resets msg to take in further values
	}

}

func ForwardQuery(peerId int, queryEmbed []float64, startPeerAddr *net.UDPAddr, genPeerConn *net.UDPConn) error {
	if len(queryEmbed) != consts.EMBED_DIM {
		return fmt.Errorf("embedding dimension mismatch: expected %v", consts.EMBED_DIM)
	}

	msg := models.MessageToPeer{QueryEmbed: queryEmbed, PeerId: peerId}
	msgBytes, err := json.Marshal(msg)
	if(err != nil){
		return fmt.Errorf("error during marshalling of msg: %v", err.Error())
	}

	n,err := genPeerConn.WriteToUDP(msgBytes, startPeerAddr)
	if(err != nil){
		return fmt.Errorf("error while writing to starting peer: %v", err.Error())
	} else {
		log.Printf("Written %v bytes to %+v", n, startPeerAddr)
	}
	return nil
}