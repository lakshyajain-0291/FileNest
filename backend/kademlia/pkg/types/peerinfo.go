package types

type PeerInfo struct {
    NodeID []byte // Permanent routing ID. used for XOR, etc in kademlia
    PeerID string // temporary ID. For relay contacts
}