package network

import (
	"encoding/json"
	"log"
)

func FindValueHandler() []byte{
	var resp []byte
	return resp
}

func PingHandler(params map[string]any) []byte{
	log.Printf("params recv to PingHandler is: %+v", params)
	// add functionality for checking all params here
	
	reqJson, _ := json.Marshal(params)
	return reqJson
}

func StoreHandler() []byte{

	var resp []byte
	return resp
}
