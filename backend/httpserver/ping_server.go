package main

import (
    "fmt"
    "log"
    "net/http"
    "os"
)

func main() {
    // Handle ping requests
    http.HandleFunc("/ping", func(w http.ResponseWriter, r *http.Request) {
        fmt.Printf("Received ping from: %s\n", r.RemoteAddr)
        fmt.Fprintln(w, "pong")
    })

    // Handle root requests
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Printf("Received request from: %s for path: %s\n", r.RemoteAddr, r.URL.Path)
        fmt.Fprintln(w, "HTTP Ping Server is running!")
    })

    // Get port from command line or use default
    port := "8080"
    if len(os.Args) > 1 {
        port = os.Args[1]
    }

    fmt.Printf("Starting HTTP ping server on port %s\n", port)
    fmt.Printf("Try: curl http://<this_device_ip>:%s/ping\n", port)
    
    log.Fatal(http.ListenAndServe(":"+port, nil))
}
