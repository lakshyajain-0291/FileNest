package peer

import (
	"final/network/RelayFinal/pkg/network"
	"final/network/RelayFinal/pkg/relay/models"

	"encoding/json"
	"fmt"
	"math/big"
)

var Peer *models.UserPeer

type RelayDist struct {
	relayID string
	dist    *big.Int
}

// func StartNode(relayMultiAddrList []string) {
// 	fmt.Println("[DEBUG] Starting Node with relayMultiAddrList:", relayMultiAddrList)

// 	var err error
// 	Peer, err = NewPeer(relayMultiAddrList, "depth")
// 	if err != nil {
// 		fmt.Println("[ERROR] Error creating peer:", err)
// 		return
// 	}

// 	ctx := context.Background()
// 	if err := Start(Peer, ctx); err != nil {
// 		log.Fatal("[FATAL] Start failed:", err)
// 	}
// 	fmt.Println("[DEBUG] Node started successfully.")
// }

// func GET(targetPeerID string, route string) ([]byte, error) {
// 	fmt.Println("[DEBUG][GET] Called with targetPeerID:", targetPeerID, " route:", route)

// 	reqparams := make(map[string]string)
// 	parts := strings.Split(route, "/")
// 	fmt.Println("[DEBUG][GET] Route split parts:", parts)

// 	params := strings.Split(parts[1], "&&")
// 	fmt.Println("[DEBUG][GET] Extracted params:", params)

// 	for i := range params {
// 		key := strings.Split(params[i], "=")[0]
// 		value := strings.Split(params[i], "=")[1]
// 		reqparams[key] = value
// 		fmt.Printf("[DEBUG][GET] Param parsed key=%s value=%s\n", key, value)
// 	}

// 	reqparams["Method"] = "GET"

// 	jsonReq, err := json.Marshal(reqparams)
// 	if err != nil {
// 		fmt.Println("[ERROR][GET] Failed to marshal req params:", err)
// 		return nil, err
// 	}
// 	fmt.Println("[DEBUG][GET] Marshaled reqparams JSON:", string(jsonReq))

// 	ctx := context.Background()
// 	GetResp, err := Send(Peer, ctx, targetPeerID, jsonReq, nil)
// 	if err != nil {
// 		fmt.Println("[ERROR][GET] Error sending request:", err)
// 		return nil, err
// 	}

// 	GetResp = bytes.TrimRight(GetResp, "\x00")
// 	fmt.Println("[DEBUG][GET] Response (trimmed):", string(GetResp))
// 	return GetResp, nil
// }

// func POST(targetPeerID string, route string, body []byte) ([]byte, error) {
// 	fmt.Println("[DEBUG][POST] Called with targetPeerID:", targetPeerID, " route:", route, " body:", string(body))

// 	ctx := context.Background()
// 	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
// 	defer cancel()

// 	reqparams := make(map[string]string)
// 	parts := strings.Split(route, "/")
// 	fmt.Println("[DEBUG][POST] Route split parts:", parts)

// 	params := strings.Split(parts[1], "&&")
// 	fmt.Println("[DEBUG][POST] Extracted params:", params)

// 	for i := range params {
// 		key := strings.Split(params[i], "=")[0]
// 		value := strings.Split(params[i], "=")[1]
// 		reqparams[key] = value
// 		fmt.Printf("[DEBUG][POST] Param parsed key=%s value=%s\n", key, value)
// 	}

// 	reqparams["Method"] = "POST"

// 	jsonReq, err := json.Marshal(reqparams)
// 	if err != nil {
// 		fmt.Println("[ERROR][POST] Failed to marshal req params:", err)
// 		return nil, err
// 	}
// 	fmt.Println("[DEBUG][POST] Marshaled reqparams JSON:", string(jsonReq))

// 	GetResp, err := Send(Peer, timeoutCtx, targetPeerID, jsonReq)
// 	if err != nil {
// 		fmt.Println("[ERROR][POST] Error sending request:", err)
// 		return nil, err
// 	}

// 	GetResp = bytes.TrimRight(GetResp, "\x00")
// 	fmt.Println("[DEBUG][POST] Response (trimmed):", string(GetResp))
// 	return GetResp, nil
// }

func ServeGetReq(paramsBytes []byte) []byte {
	fmt.Println("[DEBUG][ServeGetReq] Received params:", string(paramsBytes))

	var params map[string]any
	err := json.Unmarshal(paramsBytes, &params)
	if err != nil {
		fmt.Println("[ERROR][ServeGetReq] Failed to unmarshal params:", err)
	}
	fmt.Println("[DEBUG][ServeGetReq] Parsed params:", params)


	switch params["Route"] {
	case "find_value":
		fmt.Println("[DEBUG][ServeGetReq] Handling route: find_value")
		return network.FindValueHandler(params)

	case "ping":
		fmt.Println("[DEBUG][ServeGetReq] Handling route: ping")
		return network.PingHandler(params)

	default:
		fmt.Println("[WARN][ServeGetReq] Unknown route:", params["route"])
	}

	var resp []byte
	return resp
}

func ServePostReq(paramsBytes []byte, bodyBytes []byte) []byte {

	var params map[string]any
	err := json.Unmarshal(paramsBytes, &params)
	if err != nil {
		fmt.Println("[ERROR][ServePostReq] Failed to unmarshal params:", err)
	}

	switch params["Route"] {
	case "store":
		fmt.Println("[DEBUG][ServePostReq] Handling route: store")
		return network.StoreHandler()
		
	case "ftp":
		fmt.Println("[DEBUG][ServeSendReq] Received params:", string(paramsBytes), " body:", string(bodyBytes))
		
		var params map[string]any
		err := json.Unmarshal(paramsBytes, &params)
		if err != nil {
			fmt.Println("[ERROR][ServeSendReq] Failed to unmarshal params:", err)
		}
		fmt.Println("[DEBUG][ServeSendReq] Parsed params:", params)
		
		return network.SendHandler(params,bodyBytes)

	default:
		fmt.Println("[WARN][ServePostReq] Unknown POST route:", params["route"])
	}

	var resp []byte
	return resp
}

