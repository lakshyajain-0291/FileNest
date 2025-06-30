# UDP Messaging Client (Go)

A simple UDP-based messaging client implemented in Go. It allows two users to exchange text messages over a local network using static ports. This project demonstrates basic socket programming, CLI handling, and concurrent message receiving using goroutines.

---

## 🚀 Features

- Send and receive UDP messages over a local network.
- Static ports for predictable communication (no ephemeral ports).
- Command Line Interface (CLI) with configurable IP and port.
- Non-blocking message reception using goroutines.

---

## 📦 Requirements

- Go 1.18 or later installed
- Two terminal instances or two machines on the same local network

---

## 🛠️ Installation

1. Clone the repository:
   ```
   git clone https://github.com/Phantom0133/udp-messaging-client.git
   cd udp-messaging-client

## 💻 Usage Example
Terminal 1:
```
go run main.go -ip 127.0.0.1 -target_port 3001 -local_port 3000
```
Terminal 2:
```
go run main.go -ip 127.0.0.1 -target_port 3000 -local_port 3001
