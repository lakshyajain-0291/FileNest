package network

import (
	"encoding/json"
	_ "final/network/RelayFinal/pkg/relay/models"
	"log"
	"os"
)

func FindValueHandler(params map[string]any) []byte{
	log.Printf("params recv to PingHandler is: %+v", params)
	// add functionality for checking all params here
	
	reqJson, _ := json.Marshal(params)
	return reqJson
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

var fileBuffers = make(map[string][][]byte) // fileID -> slice of chunks
var fileMeta = make(map[string]map[string]any) // store filename/total size etc.

func SendHandler(params map[string]any, body []byte) []byte {
	log.Printf("SendHandler Called with params: %+v\n", params)

	msgType, ok := params["type"].(string)
	if !ok {
		log.Println("invalid params, no type field")
		return nil
	}

	fileID := params["fileID"].(string)

	switch msgType {
	case "START":
		// initialize buffer
		fileBuffers[fileID] = make([][]byte, 0)
		fileMeta[fileID] = params
		log.Printf("Started receiving file %s with ID %s", params["filename"], fileID)

	case "CHUNK":
		// collect chunk
		index := int(params["index"].(float64)) // JSON numbers decode as float64
		if len(fileBuffers[fileID]) <= index {
			// grow slice to fit index
			newBuf := make([][]byte, index+1)
			copy(newBuf, fileBuffers[fileID])
			fileBuffers[fileID] = newBuf
		}
		fileBuffers[fileID][index] = body
		log.Printf("Received chunk %d for file %s", index, fileID)

	case "END":
		// stitch chunks
		chunks := fileBuffers[fileID]
		var fileData []byte
		for _, c := range chunks {
			fileData = append(fileData, c...)
		}
		filename := fileMeta[fileID]["filename"].(string)
		err := os.WriteFile("./received_"+filename, fileData, 0644)
		if err != nil {
			log.Printf("Error writing file: %v", err)
		} else {
			log.Printf("File %s received and saved", filename)
		}
		// cleanup
		delete(fileBuffers, fileID)
		delete(fileMeta, fileID)
	}

	return nil
}

