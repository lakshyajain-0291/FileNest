package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"time"
)

func main() {
	p := make([]byte, 2048)

	// User input
	var ip string
	var port string
	fmt.Printf("Enter your Ipv4 address and port please: ")
	_, err := fmt.Scan(&ip, &port)
	if err != nil {
		log.Fatal(err)
	}

	bufio.NewReader(os.Stdin).ReadString('\n')
	
	// Establishing Server Connection
	addr := fmt.Sprintf("%s:%s", ip, port)
	conn, err := net.Dial("udp", addr)
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	// Sending Message
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("Please enter your message to send to the server: ")
		if !scanner.Scan() {
			break
		}
		
		query := scanner.Text()
		if query != "q" {
			fmt.Fprintf(conn, query)
			time.Sleep(time.Second)
			_, err = bufio.NewReader(conn).Read(p)
			if err == nil {
				fmt.Printf("%s\n", p)
			} else {
				fmt.Printf("Some error %v\n", err)
			}
		} else {
			fmt.Fprintf(conn, "q")
			fmt.Println("Ending conversation...")
			break
		}
	}
}
