package peer

import (
	// ...
	"bytes"
	"context"

	//"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"strings"
	"time"
)

var Peer *DepthPeer
// var globalLocalNode *models.Node
// var GlobalRT *routing.RoutingTable

type RelayDist struct {
	relayID string
	dist    *big.Int
}
func StartNode(relayMultiAddrList []string) {

	fmt.Println("Starting Node...")
	var err error
	Peer, err = NewDepthPeer(relayMultiAddrList)
	if err != nil {
		fmt.Println("Error creating peer:", err)
		return
	}

	ctx := context.Background()

	if err := Peer.Start(ctx); err != nil {
		log.Fatal(err)
	}

	// initDHT()
}

func GET(targetPeerID string ,route string) ([]byte, error) { //"/ts=123&&id=123"

	reqparams := make(map[string]string)
	parts := strings.Split(route, "/")

	params := strings.Split(parts[1], "&&")

	for i := range len(params) {
		key := strings.Split(params[i], "=")[0]
		value := strings.Split(params[i], "=")[1]

		reqparams[key] = value
	}
	reqparams["Method"] = "GET"
	jsonReq, err := json.Marshal(reqparams)
	if err != nil {
		fmt.Println("[DEBUG]Failed to get req params json")
		return nil, err
	}
	_ = jsonReq
	ctx := context.Background()

	GetResp, err := Peer.Send(ctx,targetPeerID, jsonReq, nil)
	if err != nil {
		fmt.Println("Error Sending trial get message")
	}
	GetResp = bytes.TrimRight(GetResp, "\x00")
	return GetResp, nil //this will be json bytes with resp encoded in form of resp from the server and can be used according to utility
}

func POST(targetPeerID string, route string, body []byte) ([]byte, error) {

	ctx := context.Background()
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	reqparams := make(map[string]string)
	parts := strings.Split(route, "/")
	params := strings.Split(parts[1], "&&")
	for i := range len(params) {
		key := strings.Split(params[i], "=")[0]
		value := strings.Split(params[i], "=")[1]

		reqparams[key] = value
	}
	reqparams["Method"] = "POST"

	jsonReq, err := json.Marshal(reqparams)
	if err != nil {
		fmt.Println("[DEBUG]Failed to get req params json")
		return nil, err
	}

	GetResp, err := Peer.Send(timeoutCtx, targetPeerID , jsonReq, body)

	if err != nil {
		fmt.Println("Error Sending trial get message")
	}
	GetResp = bytes.TrimRight(GetResp, "\x00")
	return GetResp, nil
}

func ServeGetReq(paramsBytes []byte) []byte {
	var params map[string]interface{}
	err := json.Unmarshal(paramsBytes, &params)
	if err != nil {
		fmt.Println(err)
	}

	switch params["route"] {
	case "find_value":
		keyStr, ok := params["ts"].(string)
		if !ok {
			fmt.Println("ts is not a string")
		}
		fmt.Printf("Timestamp to retrieve: %s", keyStr)
	//	return network.FindValueHandler(keyStr, globalLocalNode, GlobalRT)
//uncomment this acc to use
	}

	var resp []byte
	return resp

}

// func ServePostReq(addr []byte, paramsBytes []byte, bodyBytes []byte) []byte {
// 	fmt.Println("Serving Post Request")

// 	var params map[string]interface{}
// 	if err := json.Unmarshal(paramsBytes, &params); err != nil {
// 		fmt.Println("Failed to unmarshal params:", err)
// 		return nil
// 	}

// 	route, ok := params["route"].(string)
// 	if !ok {
// 		fmt.Println("route param missing or not string")
// 		return nil
// 	}

// 	pubipStr := string(addr)
// 	ip := strings.Split(pubipStr, ":")[0]
// 	port := strings.Split(pubipStr, ":")[1]
// 	fmt.Println("IP:", ip, "Port:", port)

// 	var body map[string]interface{}
// 	if err := json.Unmarshal(bodyBytes, &body); err != nil {
// 		fmt.Println("Failed to unmarshal body:", err)
// 		return nil
// 	}
// 	fmt.Println("Body:", body)

// 	switch route {
// 	case "ping":
// 		return network.HandlePing(ip, port, body, globalLocalNode, GlobalRT)

// 	case "store":
// 		var msgCert models.MsgCert
// 		jsonBytes, _ := json.Marshal(body)
// 		if err := json.Unmarshal(jsonBytes, &msgCert); err != nil {
// 			fmt.Println("Error unmarshaling into MsgCert:", err)
// 			return nil
// 		}
// 		return network.StoreHandler(msgCert, globalLocalNode, GlobalRT)

// 	case "find_node":
// 		keyStr, ok := body["node_id"].(string)
// 		if !ok || keyStr == "" {
// 			fmt.Println("find_node error: node_id is missing or not a string")
// 			errResp := map[string]interface{}{"error": "node_id is missing or not a string"}
// 			resp, _ := json.Marshal(errResp)
// 			return resp
// 		}
// 		keyPubKeyStr, ok := body["public_key"].(string)
// 		if !ok || keyPubKeyStr == "" {
// 			fmt.Println("find_node error: public_key is missing or not a string")
// 			errResp := map[string]interface{}{"error": "public_key is missing or not a string"}
// 			resp, _ := json.Marshal(errResp)
// 			return resp
// 		}
// 		// Compose a body map as expected by FindNodeHandler
// 		findNodeBody := map[string]interface{}{
// 			"node_id":    keyStr,
// 			"public_key": keyPubKeyStr,
// 		}
// 		return network.FindNodeHandler(ip, port, findNodeBody, globalLocalNode, GlobalRT)

// 	case "delete":
// 		var repCert models.ReportCert
// 		jsonBytes, _ := json.Marshal(body)
// 		if err := json.Unmarshal(jsonBytes, &repCert); err != nil {
// 			fmt.Println("Error unmarshaling into ReportCert:", err)
// 			return nil
// 		}
// 		return network.DeleteHandler(&repCert, globalLocalNode, GlobalRT)

// 	default:
// 		fmt.Println("Unknown POST route:", route)
// 		return nil
// 	}
// }