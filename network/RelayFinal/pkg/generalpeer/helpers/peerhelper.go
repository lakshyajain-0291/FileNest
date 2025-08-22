package helpers

import (
	"encoding/json"
	"log"
	"network/pkg/generalpeer/models"

	"github.com/libp2p/go-libp2p/core/network"
)

func Decoder(s network.Stream, msgChan chan models.Message){
    var msg models.Message
    if err := json.NewDecoder(s).Decode(&msg); err != nil {
        log.Printf("Decode error: %v", err)
        s.Reset()
        return
    }
    s.Close()
    msgChan <- msg
}
