package functions

import (
	"backend/models"
	"encoding/json"
	"errors"
)

const EMBED_DIM = 128

func ParseMessage(data []byte) (*models.Message, error) {
	var msg models.Message

	err := json.Unmarshal(data, &msg) //stores the data in msg
	if err != nil {
		return nil, err
	}

	switch msg.Source {
	case "user": //checks if the msg is from user
		if len(msg.QueryEmbed) != EMBED_DIM { // checks for the query_embed dimension matches the required dimension
			return nil, errors.New("query embed is invalid(dimension do not match)")
		}
		return &msg, nil
	case "peer": // checks if the msg is from peer
		if len(msg.Embed) != EMBED_DIM { //checks for the embed dimension matches the required dimension
			return nil, errors.New("embed vector is invalid(dimensions do not match)")
		}
		if msg.PeerID < 0 { // checks if the peer ID is valid or not
			return nil, errors.New("invalid peer ID")
		}
		if msg.FileMetadata.Name == "" { //checks for empty files
			return nil, errors.New("file name is required")
		}

		return &msg, nil
	}
	return nil, nil
}
