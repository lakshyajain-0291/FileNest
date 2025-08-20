package main

// relay addr: /ip4/10.210.211.204/tcp/10000/ws/p2p/12D3KooWBiSerxkUg4HYLXNUGgn5XQyCGem3T63E762EmiFd47kq
// self addr: /dns4/filenest-q5fr.onrender.com/tcp/10000/wss/p2p/12D3KooWBiSerxkUg4HYLXNUGgn5XQyCGem3T63E762EmiFd47kq
import (
	"context"
	"log"
	"relay/peer"
)

func main(){
	relayAddrs := []string{"/dns4/filenest-q5fr.onrender.com/tcp/443/wss/p2p/12D3KooWAjVK3gBkC2UrCbwL4PH26PPU6TDRdbsSGx9Luvt3uFTx"}
	p,err := peer.NewDepthPeer(relayAddrs)
	if(err != nil){
		log.Printf("Error on NewDepthPeer: %v", err.Error())
	}
	ctx := context.Background()
	p.Start(ctx)

	pids := p.GetConnectedPeers()
	for pid := range pids{
		log.Printf("PIDs connected addrs are- %+v\n", pid)
	}

	select{}
}