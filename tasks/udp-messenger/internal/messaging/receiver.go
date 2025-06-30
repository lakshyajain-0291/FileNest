package messaging

import (
	"fmt"
	"net"
)

/*
StartReceiver launches a goroutine that listens for incoming UDP messages
on the provided connection and prints them to the console.
*/
func StartReceiver(conn *net.UDPConn) {
	go func() {
		buffer := make([]byte, 1024)
		for {
			n, addr, err := conn.ReadFromUDP(buffer)
			if err != nil {
				fmt.Println("Error receiving message:", err)
				continue
			}
			fmt.Printf("Received from %s: %s\n", addr.String(), string(buffer[:n]))
		}
	}()
}
