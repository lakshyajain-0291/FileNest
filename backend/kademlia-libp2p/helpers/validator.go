package helper

// receives target node id as input, checks if the target node id is the same as the peer's node id. 
// if yes then you move onto execution.go
// if not then use kademlia to contact the next peer. 
// call this function whenever you receive an incoming request. 