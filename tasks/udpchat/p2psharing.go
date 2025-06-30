package main

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"strings"
)

func main() {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter listener port: ")
	peerport, _ := reader.ReadString('\n')
	peerport = strings.TrimSpace(peerport)

	listener, err := net.ListenPacket("udp", ":"+peerport)
	if err != nil {
		fmt.Println("Error listening ", err)
		return
	}
	defer listener.Close()

	go Startlisten(listener)

	fmt.Print("Enter peer address: ")
	peerAddrStr, _ := reader.ReadString('\n')
	peerAddrStr = strings.TrimSpace(peerAddrStr)
	peerUDPAddr, err := net.ResolveUDPAddr("udp", peerAddrStr)
	if err != nil {
		fmt.Println("Could not resolve peer address: ", err)
		return
	}

	for {
		fmt.Printf("You: ")
		msg, _ := reader.ReadString('\n')
		listener.WriteTo([]byte(msg), peerUDPAddr)
	}
}

func Startlisten(listener net.PacketConn) {
	buf := make([]byte, 1024)
	for {
		n, addr, err := listener.ReadFrom(buf)
		if err != nil {
			fmt.Println("Error reading: ", err)
			continue
		}
		if udpAddr, ok := addr.(*net.UDPAddr); ok {
			fmt.Printf("...\nFrom [%s:%d]: %s\n", udpAddr.IP, udpAddr.Port, string(buf[:n]))
		} else {
			fmt.Printf("...\nFrom [%s]: %s\n", addr.String(), string(buf[:n]))
		}
		fmt.Printf("You: ")
	}
}
