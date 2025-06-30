package main

import (
	"bufio"
	"flag"
	"fmt"
	"net"
	"os"
	"strings"
)

func main() {
	targetIP := flag.String("ip", "127.0.0.1", "Target IP Address")
	targetPort := flag.Int("target_port", 3000, "Target Port")
	localPort := flag.Int("local_port", 3001, "Local Port")
	flag.Parse()

	localAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf(":%d", *localPort))
	if err != nil {
		fmt.Println("Failed to resolve local address:", err)
		return
	}

	conn, err := net.ListenUDP("udp", localAddr)
	if err != nil {
		fmt.Println("Failed to bind to UDP port:", err)
		return
	}

	defer conn.Close()

	go receiveMessage(conn)

	fmt.Printf("Sending messages to %s:%d\n", *targetIP, *targetPort)
	targetAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", *targetIP, *targetPort))
	if err != nil {
		fmt.Println("Failed to resolve target address:", err)
		return
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		text, _ := reader.ReadString('\n')
		text = strings.TrimSpace(text)

		if text == "" {
			continue
		}

		_, err := conn.WriteToUDP([]byte(text), targetAddr)
		if err != nil {
			fmt.Println("Failed to send message:", err)
		}
	}
}

func receiveMessage(conn *net.UDPConn) {
	buffer := make([]byte, 1024)
	for {
		n, addr, err := conn.ReadFromUDP(buffer)
		if err != nil {
			fmt.Println("Error receiving message:", err)
			continue
		}
		fmt.Printf("\nReceived from %s: %s\n", addr.String(), string(buffer[:n]))
	}
}
