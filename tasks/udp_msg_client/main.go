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
	localPort := flag.String("local-port", "8080", "Local UDP port")
	targetPort := flag.String("target-port", "8081", "Target UDP port")
	flag.Parse()

	localAddr, err := net.ResolveUDPAddr("udp", "127.0.0.1:"+*localPort)
	if err != nil {
		fmt.Println("Failed to resolve the local address:", err)
		return
	}
	targetAddr, err := net.ResolveUDPAddr("udp", "127.0.0.1:"+*targetPort)
	if err != nil {
		fmt.Println("Failed to resolve the target address:", err)
		return
	}

	conn, err := net.ListenUDP("udp", localAddr)
	if err != nil {
		fmt.Println("Failed to bind to the UDP port:", err)
		return
	}

	defer conn.Close()

	fmt.Printf("Binding to 127.0.0.1:%s\n", *localPort)
	fmt.Println(">>Type your message & press Enter (type 'exit' to quit)")
	fmt.Printf(">>Target: %s\n", targetAddr.String())

	done := make(chan struct{})

	// Receiver goroutine
	go func() {
		buf := make([]byte, 1024)
		for {
			select {
			case <-done:
				return
			default:
				n, addr, err := conn.ReadFromUDP(buf)
				if err != nil {
					continue
				}
				msg := strings.TrimSpace(string(buf[:n]))
				// Avoid printing the dummy exit message to self
				if msg == "##exit##" && addr.String() == localAddr.String() {
					return
				}
				fmt.Printf(">> Received from %s: %s\n", addr, msg)
			}
		}
	}()

	// Sender loop
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print(">> ")
		scanner.Scan()
		text := scanner.Text()

		if strings.ToLower(text) == "exit" {
			fmt.Println("Exiting...")
			close(done)
			// Send dummy message to self to unblock ReadFromUDP
			conn.WriteToUDP([]byte("##exit##"), localAddr)
			break
		}

		conn.WriteToUDP([]byte(text), targetAddr)
	}
}
