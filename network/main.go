package main

import (
	"context"
	"encoding/json"
	"final/network/RelayFinal/pkg/network"
	"final/network/RelayFinal/pkg/relay/helpers"
	"final/network/RelayFinal/pkg/relay/peer"
	"flag"
	"log"
)



func main() {
	pid := flag.String("pid", "", "Target peer ID to send request to (optional)")
	flag.Parse()

	relayAddrs, err := helpers.GetRelayAddrFromMongo()
	if err != nil {
		log.Printf("Error during get relay addrs: %v", err.Error())
	}
	log.Printf("relayAddrs in Mongo: %+v\n", relayAddrs)

	// Initialize peer
	ctx := context.Background()
	p, err := peer.NewPeer(relayAddrs, "user")
	if err != nil {
		log.Fatalf("Error creating %s peer: %v", "user", err)
	}
	peer.Start(p, ctx)
	log.Printf("Started %s peer successfully", "user")

	params := struct {
		Type string
		Route string
		Filename string
		PeerId string
	}{
		Type: "POST",
		Route: "ftp",
		Filename: "car_recv",
		PeerId: *pid,
	}
	paramsJson,_ := json.Marshal(params)

	imgBytes, _ := network.ImageToBytes("./imgs/car.jpeg", "jpeg")
	// encoded := base64.StdEncoding.EncodeToString(imgBytes)

	if *pid != "" {
		if _,err := peer.Send(p,ctx,*pid,paramsJson,imgBytes); err != nil {
			log.Printf("Send failed: %v", err)
		}
	}

	select {}
}
