package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"
	"udp-messenger/internal/messaging"
)

/*
main is the entry point of the UDP messenger application.
It parses command-line flags for target and local ports/IP,
creates a UDP socket, and starts the receiver and sender routines.
*/
func main() {
	targetIP := flag.String("target-ip", "127.0.0.1", "Target IP address")
	targetPort := flag.Int("target-port", 0, "Target port")
	localPort := flag.Int("local-port", 0, "Local port to bind to")
	flag.Parse()

	if *targetPort == 0 || *localPort == 0 {
		fmt.Println(">>Usage: --local-port <port> --target-port <port> --target-ip <ip>")
		flag.PrintDefaults()
		os.Exit(1)
	}
	// Create UDP socket bound to the specified local port.
	conn, err := messaging.CreateUDPSocket(*localPort)
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	// Resolve the target UDP address.
	targetAddrStr := *targetIP + ":" + strconv.Itoa(*targetPort)
	targetAddr, err := net.ResolveUDPAddr("udp", targetAddrStr)
	if err != nil {
		log.Fatal(">>Invalid target address:", err)
	}

	// Start the receiver and sender goroutines.
	messaging.StartReceiver(conn)
	messaging.StartSender(conn, targetAddr)

	// Block forever to keep the main goroutine alive.
	select {}
}
