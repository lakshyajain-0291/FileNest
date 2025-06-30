package messaging

import (
	"fmt"
	"net"
	"strconv"
)

/*
CreateUDPSocket binds to a static UDP port (provided by localPort)
and returns the UDP connection. It returns an error if the address
cannot be resolved or the socket cannot be bound.
*/
func CreateUDPSocket(localPort int) (*net.UDPConn, error) {
	addrStr := "127.0.0.1:" + strconv.Itoa(localPort)
	fmt.Println("Binding to", addrStr)

	udpAddr, err := net.ResolveUDPAddr("udp", addrStr)
	if err != nil {
		return nil, fmt.Errorf(">>failed to resolve address: %w", err)
	}

	conn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		return nil, fmt.Errorf(">>failed to bind to port: %w", err)
	}

	return conn, nil
}
