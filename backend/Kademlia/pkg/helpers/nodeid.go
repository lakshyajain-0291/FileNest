package helpers

import (
    "crypto/sha1"
)

// HashNodeID normalizes any input to a fixed-length NodeID
func HashNodeID(input []byte) []byte {
    hash := sha1.Sum(input)
    return hash[:] // Always returns 20 bytes (160 bits) - SHA1 hash length
}

// HashNodeIDFromString generates a NodeID from a string input - ADDED FUNCTION
func HashNodeIDFromString(input string) []byte {
    return HashNodeID([]byte(input))
}

// HashNodeIDShort - Since SHA1 is already 20 bytes, this is the same as HashNodeID
func HashNodeIDShort(input []byte) []byte {
    hash := sha1.Sum(input)
    return hash[:] // Returns 20 bytes (same as above since SHA1 is 20 bytes)
}
