package main

import (
	"general-peer/pkg/models"
	"log"
)

func main() {
	msgChan := make(chan models.Message)
	mlChan := make(chan models.ClusterWrapper)

	// ML Receive Loop
	go func() {
		log.Println("[NET] Listening for ML messages...")
		for {
			msg := <-mlChan
			log.Printf("[NET] Received ML message: %+v", msg)
		}
	}()

	// Network recieve loop
	go func() {
		log.Println("[NET] Listening for peer messages...")
		for {
			msg := <-msgChan
			log.Printf("[NET] Received Peer message: %+v", msg)
		}
	}()

	select {}
}
