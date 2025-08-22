package helpers

import (
	"crypto/rand"
	"errors"
	"kademlia/pkg/identity"
	"log"
	"math/big"

	"github.com/libp2p/go-libp2p/core/peer"
)

// Parser to validate BootStrapAddr
func ParseBootstrapAddr(addr string) (peer.AddrInfo, error) {
    maddr, err := peer.AddrInfoFromString(addr)
    if err != nil {
        return peer.AddrInfo{}, errors.New("invalid bootstrap node multiaddr")
    }
    return *maddr, nil
}

func XORDistance(a, b []byte) *big.Int{
	if len(a) != len(b) {
		panic("IDs must be the same length")
	}

	dist := make([]byte, len(a))
	for i := range a {
		dist[i] = a[i] ^ b[i]
	}

	return new(big.Int).SetBytes(dist)
}

func BucketIndex(selfID, otherID []byte) int {
    if len(selfID) != len(otherID) {
        panic("IDs must be the same length")
    }

    for byteIndex := range selfID {
        xorByte := selfID[byteIndex] ^ otherID[byteIndex]

        if xorByte != 0 { // first differing bit
            // Find position of first set bit in this byte
        
			for bitPos := range 8 {
                if (xorByte & (0x80 >> bitPos)) != 0 {
                   return (len(selfID)-byteIndex-1)*8 + (7 - bitPos)
                }
            }
        }
    }
    return -1 // identical IDs → no bucket
}

func RandomNodeID() []byte {
	id := make([]byte, identity.NodeIDBytes)
	if _, err := rand.Read(id); err != nil {
		log.Fatalf("failed to generate random NodeID: %v", err)
	}
	return id
}