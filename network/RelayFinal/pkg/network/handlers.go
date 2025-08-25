package network

import (
	"encoding/json"
	"log"
	"network/pkg/relay/models"
)

func FindValueHandler() []byte{
	var resp []byte
	return resp
}

func PingHandler(params map[string]any, body map[string]any) []byte{
	if(body["sender_addr"] == ""){
		log.Printf("Invalid Sender addr: %v", body["sender_addr"])
		return nil;
	}
	bodyJson, _ := json.Marshal(body)
	var req models.PingRequest
	json.Unmarshal(bodyJson,&req)
	reqJson, _ := json.Marshal(req)
	return reqJson
}

func StoreHandler() []byte{

	var resp []byte
	return resp
}
