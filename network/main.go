package main

import (
	"context"
	"encoding/json"
	"final/network/RelayFinal/pkg/network"
	"final/network/RelayFinal/pkg/relay/helpers"
	"final/network/RelayFinal/pkg/relay/models"
	"final/network/RelayFinal/pkg/relay/peer"
	"flag"
	"log"
	"time"
)

const ChunkSize = 32 * 1024 // 32KB per chunk

func sendFile(p *models.UserPeer, ctx context.Context, pid string, filePath string) error {
	// Convert image to bytes
	bodyBytes, err := network.ImageToBytes(filePath, "jpeg")
	if err != nil {
		return err
	}

	total := len(bodyBytes)
	fileID := "123" // you can replace with UUID

	// 1. Send START
	startParams := models.PingRequest{
		Type:           "SEND",
		Route:          "ftp",
		ReceiverPeerID: pid,
		Timestamp:      time.Now().Unix(),
	}
	startParamsJSON, _ := json.Marshal(startParams)

	meta := map[string]any{
		"type":     "START",
		"fileID":   fileID,
		"filename": "car.jpeg",
		"total":    total,
	}
	metaJSON, _ := json.Marshal(meta)

	if _, err := peer.Send(p, ctx, pid, startParamsJSON, metaJSON); err != nil {
		return err
	}

	for i := 0; i*ChunkSize < total; i++ {
		start := i * ChunkSize
		end := start + ChunkSize
		if end > total {
			end = total
		}

		chunkMeta := map[string]any{
			"type":   "CHUNK",
			"fileID": fileID,
			"index":  i,
		}
		chunkMetaJSON, _ := json.Marshal(chunkMeta)

		if _, err := peer.Send(p, ctx, pid, chunkMetaJSON, bodyBytes[start:end]); err != nil {
			return err
		}
		log.Printf("Sent chunk %d (%d - %d)", i, start, end)
	}

	// 3. Send END
	endMeta := map[string]any{
		"type":   "END",
		"fileID": fileID,
	}
	endMetaJSON, _ := json.Marshal(endMeta)

	if _, err := peer.Send(p, ctx, pid, endMetaJSON, nil); err != nil {
		return err
	}

	return nil
}

func main() {
	pid := flag.String("pid", "", "Target peer ID to send request to (optional)")
	flag.Parse()

	relayAddrs, err := helpers.GetRelayAddrFromMongo()
	if err != nil {
		log.Printf("Error during get relay addrs: %v", err.Error())
	}
	log.Printf("relayAddrs in Mongo: %+v\n", relayAddrs)

	// Initialize peer
	ctx := context.Background()
	p, err := peer.NewPeer(relayAddrs, "user")
	if err != nil {
		log.Fatalf("Error creating %s peer: %v", "user", err)
	}
	peer.Start(p, ctx)
	log.Printf("Started %s peer successfully", "user")

	// If pid is provided, send file
	if *pid != "" {
		if err := sendFile(p, ctx, *pid, "./imgs/car.jpeg"); err != nil {
			log.Printf("Send failed: %v", err)
		}
	}

	select {}
}
