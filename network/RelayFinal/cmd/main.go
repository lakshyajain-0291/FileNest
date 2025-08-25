package main

import (
	"context"
	"encoding/json"
	"log"
	genmodels "network/pkg/generalpeer/models"
	"network/pkg/generalpeer/ws"
	"network/pkg/relay/helpers"
	relaymodels "network/pkg/relay/models"
	"network/pkg/relay/peer"
	"time"
)

func main(){
	var mlChan chan genmodels.ClusterWrapper

	peerTransport := ws.NewWebSocketTransport(":8080")
	mlTransport := ws.NewWebSocketTransport(":8081")
	defer func () {
		mlTransport.Close()
		peerTransport.Close()
	}()

	go func(){
		if err := mlTransport.StartMLReceiver(mlChan); err != nil {
			log.Printf("ML Receiver error: %v", err)
		}
	}()

	go func() {
		log.Println("[NET] Listening for ML messages...")
		for {
			msg := <-mlChan
			log.Printf("[NET] Received ML message: %+v", msg)
		}
	}()
	
	relayAddrs, err := helpers.GetRelayAddrFromMongo()
	if(err != nil){
		log.Printf("Error during get relay addrs: %v", err.Error())
	}
	log.Printf("relayAddrs in Mongo: %+v\n", relayAddrs)

	p,err := peer.NewDepthPeer(relayAddrs)
	if(err != nil){
		log.Printf("Error on NewDepthPeer: %v\n", err.Error())
	}

	ctx := context.Background()
	peer.Start(p,ctx)
		
	pid := "12D3KooWAK9vNDzZhftJepCDBdf9gsAcidnQj6Kpv4k6nCanirmZ"

	req := relaymodels.ReqFormat{
		Type: "GET",
		PeerID: pid,
	}
	reqJson, _ := json.Marshal(req);

	body := struct {
		Route string
		Ts time.Time
	}{	
		"find_value",
		time.Now(),
	}
	bodyJson, _ := json.Marshal(body);

	resp, err := peer.Send(p,ctx, pid, reqJson, bodyJson)
	if(err != nil){
		log.Println(err.Error())
	}

	var respDec any;
	json.Unmarshal(resp, &respDec)
	log.Printf("Response: %+v", respDec)

	select{}
}
