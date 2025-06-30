package main

import (
	"bufio" //buffered I/O to read from terminal
	"flag"  // command-line flag parsing
	"fmt"
	"net" //networking for UDP(User Datagram Protocol) sockets
	"os"
	"os/signal" //to get os singals like ctrl + c for this one
	"strings"
	"syscall"
)

func main() {
	localPort := flag.Int("local-port", 8000, "Local UDP port to bind")                        //port to bind to locally
	targetPort := flag.Int("target-port", 8001, "Target UDP port to send messages to")         //target port to send the messages to
	targetIP := flag.String("target-ip", "127.0.0.1", "Target IP address to send messages to") //target IP address to send the messages to
	flag.Parse()

	targetAddr := net.JoinHostPort(*targetIP, fmt.Sprintf("%d", *targetPort)) //joining the target Ip and port in the form of host:port

	localAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf(":%d", *localPort)) //Parsing the text form of an address and turning it into a structured data object we can give to the kernel
	if err != nil {
		fmt.Println("Failed to resolve local address:", err)
		return
	}

	remoteAddr, err := net.ResolveUDPAddr("udp", targetAddr)
	if err != nil {
		fmt.Println("Failed to resolve remote address:", err)
		return
	}

	conn, err := net.ListenUDP("udp", localAddr) //binding the UDP socket to the local address
	if err != nil {
		fmt.Println("Failed to bind UDP socket:", err)
		return
	}
	defer conn.Close() //ensuring the socket closes when the main exits

	fmt.Printf("Binding to %s\n", localAddr.String())
	fmt.Printf("Target: %s\n", remoteAddr.String())
	fmt.Println(">> Type your message and press Enter (Ctrl+C to quit)")

	sigChan := make(chan os.Signal, 1)                    //channel to handle ctrl + c
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM) //here, os.Interrupt is for Ctrl+C and syscall.SIGTERM is for termination signals, and signal.notify tells go to send these to sigchan channnel

	go func() { //concurrent goroutine to wait for udp datagram input, also defined and called the function at once
		buf := make([]byte, 1024) //buffer holding an array of 1024 bytes to read incoming messages
		for {
			n, addr, err := conn.ReadFromUDP(buf) //putting the message in buf, n is bytes here
			if err != nil {
				fmt.Println("Error receiving:", err)
				continue
			}
			fmt.Printf("\n>> Received from %s: %s\n", addr.String(), strings.TrimSpace(string(buf[:n]))) //buf[:n] slices the part to show what has been received, string converts bytes into go string, and trimspace takes care of new lines and extra space
			fmt.Print(">> ")
		}
	}()

	scanner := bufio.NewScanner(os.Stdin) //scanner to read input from teh terminal

	go func() { //goroutine for sending messages, concurrency is used so as we don't block the rest of the program(from getting the messages)
		for {
			fmt.Print(">> ")
			if !scanner.Scan() { //waiting till enter
				break
			}
			text := strings.TrimSpace(scanner.Text()) //helps in removing extra whitespaces
			if text == "" {                           //to handle if we press enter without typing anything
				fmt.Println("Empty message, please type something.")
				continue
			}
			_, err := conn.WriteToUDP([]byte(text), remoteAddr) //converting my string to a slice of bytes and sending to remoteaddr
			if err != nil {
				fmt.Println("Error sending:", err)
			}
		}
	}()

	<-sigChan //waiting for ctrl + c/termination signal
	fmt.Println("\nExiting...")
}
