package main

import (
	"fmt"
	"log"
	"net"
)


func sendResponse(conn *net.UDPConn, addr *net.UDPAddr) {
    _,err := conn.WriteToUDP([]byte("From server: Hello I got your message "), addr)
    if err != nil {
        fmt.Printf("Couldn't send response %v", err)
		log.Fatal((err))
    }
}


func main() {
    p := make([]byte, 2048)
    addr := net.UDPAddr{
        Port: 1234,
        IP: net.ParseIP("127.0.0.1"),
    }
    ser, err := net.ListenUDP("udp", &addr)
    if err != nil {
        fmt.Printf("Some error %v\n", err)
        return
    }
    for {
        n,remoteaddr,err := ser.ReadFromUDP(p)
		message := string(p[:n])
		if message == "q"{
			fmt.Println("Ending conversation...")
			break
		}else{
			fmt.Printf("Read a message from %v %s \n", remoteaddr, message)
			if err !=  nil {
				fmt.Printf("Some error  %v", err)
				continue
			}
			sendResponse(ser, remoteaddr)
		}
    }
	ser.Close()
}