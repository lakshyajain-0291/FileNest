package helpers

import (
    "crypto/sha256"
)

// HashNodeID normalizes any input to a fixed-length NodeID
func HashNodeID(input []byte) []byte {
    hash := sha256.Sum256(input)
    return hash[:] // Always returns 32 bytes
}

// Alternative: Use shorter hash if you prefer
func HashNodeIDShort(input []byte) []byte {
    hash := sha256.Sum256(input)
    return hash[:20] // Use first 20 bytes for shorter NodeIDs
}
