package main

// relay addr: /ip4/10.210.211.204/tcp/10000/ws/p2p/12D3KooWBiSerxkUg4HYLXNUGgn5XQyCGem3T63E762EmiFd47kq
// self addr: /dns4/filenest-q5fr.onrender.com/tcp/10000/wss/p2p/12D3KooWBiSerxkUg4HYLXNUGgn5XQyCGem3T63E762EmiFd47kq
import (
	"context"
	"encoding/json"
	"log"
	"relay/helpers"
	"relay/models"
	"relay/peer"
)


func main(){
	relayAddrs, err := helpers.GetRelayAddrFromMongo()
	if(err != nil){
		log.Printf("Error during get relay addrs: %v", err.Error())
	}
	log.Printf("relayAddrs: %+v", relayAddrs)
	
	p,err := peer.NewDepthPeer(relayAddrs)
	if(err != nil){
		log.Printf("Error on NewDepthPeer: %v", err.Error())
	}
	ctx := context.Background()
	peer.Start(p,ctx)
	
	pid := "12D3KooWM7Re1uneHMYdhZr9S5ADMfCwyPdLz8Lfe7icWCvnAJrP"
	// reqParams := {
	// 	""
	// }
	req := models.ReqFormat{
		Type: "GET",
		PeerID: pid,
	}
	reqJson, _ := json.Marshal(req);
	resp, err := peer.Send(p,ctx, pid, reqJson, nil)
	if(err != nil){
		log.Println(err.Error())
	}

	var respDec any;
	json.Unmarshal(resp, &respDec)
	log.Printf("Response: %+v", respDec)

	select{}
}