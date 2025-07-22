package main

import (
    findvalue "dht/RPC/find_value"
    "fmt"
    "log"
    "os"
    "os/signal"
    "syscall"

    "github.com/libp2p/go-libp2p"
    "github.com/libp2p/go-libp2p/p2p/transport/tcp"
)

func main() {
    host, err := libp2p.New(libp2p.Transport(tcp.NewTCPTransport))

    if err != nil {
        log.Fatal(err)
        // return err
    }
    defer host.Close()

    // Register a stream handler to intercept messages
    host.SetStreamHandler("/jsonmessages/1.0.0", findvalue.HandleJSONMessages)

    fmt.Printf("Host ID: %s\n", host.ID())
    fmt.Printf("Listening on: %v\n", host.Addrs())

    // Keep running
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
    <-sigCh
}
