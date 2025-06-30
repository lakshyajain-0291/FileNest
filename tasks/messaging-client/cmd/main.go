package main

import (
	"flag"
	"log"
	"net"

	"github.com/yugdalwadi/udp-messenger/pkg/messenger"
)

func main(){
	targetPort := flag.Int("tport", 8001, "Port to send message to")
	targetIp := flag.String("tip", "127.0.0.1", "IP to send msg to")
	senderPort := flag.Int("sport", 8002, "Port to send message from")
	senderIp := flag.String("sip", "127.0.0.1", "IP to send msg from")
	flag.Parse()

	senderAddr := &net.UDPAddr{IP:net.ParseIP(*senderIp), Port: *senderPort}
	targetAddr := &net.UDPAddr{IP:net.ParseIP(*targetIp), Port: *targetPort}

	Conn, err := net.ListenUDP("udp", senderAddr) // Starts a UDP Conn on given addr
	if(err != nil){
		log.Fatalf("Connection to server couldn't be created. With err: %v\n", err.Error())
	}

	log.Printf("Sender: %v Target: %v", *senderPort, *targetPort)
	messenger.HandleMessages(Conn, targetAddr)
	select{}
}