package messaging

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"strings"
)

/*
StartSender launches a goroutine that reads user input from stdin
and sends it as UDP messages to the specified target address.
Typing 'exit' will terminate the program.
*/
func StartSender(conn *net.UDPConn, targetAddr *net.UDPAddr) {
	go func() {
		reader := bufio.NewReader(os.Stdin)
		fmt.Println(">>Type your message and press Enter (type 'exit' to quit)")
		fmt.Printf(">>Target: %s:%d\n", targetAddr.IP, targetAddr.Port)

		for {
			fmt.Print(">> ")
			text, err := reader.ReadString('\n')
			if err != nil {
				fmt.Println(">>Error reading input:", err)
				continue
			}

			text = strings.TrimSpace(text)
			if text == "exit" {
				fmt.Println(">>Exiting sender.")
				os.Exit(0)
			}

			_, err = conn.WriteToUDP([]byte(text), targetAddr)
			if err != nil {
				fmt.Println(">>Error sending message:", err)
			}
		}
	}()
}
