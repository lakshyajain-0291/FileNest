package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"strings"
	"sync"
	"time"
)


func listenmsg(listener *net.UDPConn, wg *sync.WaitGroup) {
	defer wg.Done()
	listen := make([]byte, 2048)
	for {
		n, remoteaddr, err := listener.ReadFromUDP(listen)
		if err != nil {
			log.Fatal(err)
		}
		message := string(listen[:n])
		fmt.Printf("Read a message from %v %s \n", remoteaddr, message)
	}
}

func sendmsg(peerAddr *net.UDPAddr, conn *net.UDPConn, wg *sync.WaitGroup) {
	defer wg.Done()
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("Please enter your message: ")
		if !scanner.Scan() {
			break
		}

		message := strings.TrimSpace(scanner.Text())
		
		if message != "" {
			_, err := conn.WriteToUDP([]byte(message), peerAddr)
			if err != nil {
				fmt.Printf("Error sending message: %v\n", err)
			} else {
				fmt.Printf("Sent: %s\n", message)
			}
		}

		time.Sleep(100 * time.Millisecond)
	}
}

func main() {
	// Get local address
    myAddr := &net.UDPAddr{
        Port: 4321,
        IP:   net.ParseIP("127.0.0.1"),
    }
    
    // Create UDP listener
    listener, err := net.ListenUDP("udp", myAddr)
    if err != nil {
        log.Fatalf("Failed to create UDP listener: %v", err)
    }
    
    fmt.Printf("Listening on %s\n", myAddr.String())
    
    // Get peer address
    var ip, port string
    fmt.Print("Enter peer's IP address: ")
    fmt.Scan(&ip)
    fmt.Print("Enter peer's port: ")
    fmt.Scan(&port)
    
    peerAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%s", ip, port))
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Peer address: %s\n", peerAddr.String())
    
    // Use WaitGroup to coordinate goroutines
    var wg sync.WaitGroup
    wg.Add(2)
    
    // Start listening and sending goroutines
    go listenmsg(listener, &wg)
    go sendmsg(peerAddr, listener, &wg)
    
    // Wait for both goroutines to finish
    wg.Wait()
    fmt.Println("Program ended")

}
