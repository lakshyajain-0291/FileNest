package messenger

import (
	"bufio"
	"log"
	"net"
	"os"
)

func HandleMessages(Conn *net.UDPConn, targetAddr *net.UDPAddr){
	go SendMsg(Conn, targetAddr)
	go RecieveMsg(Conn, targetAddr)
}

func SendMsg(Conn *net.UDPConn, targetAddr *net.UDPAddr){
	reader := bufio.NewReader(os.Stdin) // reads from stdin

	for {
		msg,err := reader.ReadString('\n') // \n is a delimiter
		if(err != nil){
			log.Printf("Error while reading msg: %v", err.Error())
		}
		Conn.WriteToUDP([]byte(msg), targetAddr)
	}
}

func RecieveMsg(Conn *net.UDPConn, targetAddr *net.UDPAddr){
	msgBuffer := make([]byte, 1024)

	for {
		n, senderAddr, err := Conn.ReadFromUDP(msgBuffer)
		if(err != nil){
				log.Printf("Error while reading msg: %v", err.Error())
			}
		msgBuffer[n-1] = 0 // Removes the \n
		log.Printf("%v recieved from %v:%v", string(msgBuffer), senderAddr.IP,senderAddr.Port)
		msgBuffer = make([]byte, 1024) // Resets the buffer
	}
}
