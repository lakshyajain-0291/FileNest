# 📖 UDP Messaging Client (Go)

A simple command-line UDP messaging client implemented in Go that allows two instances to send and receive messages over a local network using static ports.

---

## 📌 Features

- 📡 Bidirectional text messaging over UDP
- 🖥️ Command-line interface for message input/output
- ⚙️ Customizable local and target ports via CLI flags
- 🛑 Clean and graceful shutdown via `exit` command
- ✅ No ephemeral ports — uses static ports for both sending and receiving
- 📄 Lightweight and dependency-free (only Go standard library)

---

## 📦 Requirements

- [Go](https://golang.org/dl/) 1.18+

---

## 📂 Project Structure
.
├── go.mod 
├── main.go
└── README.md


---

## 🚀 How to Run

### 1️⃣ Build / Run the client

Open **two terminal windows** (for two instances).

In **Terminal 1**:
```bash
go run main.go -local-port 8000 -target-port 8001
```

In **Terminal 2**:
```bash
go run main.go -local-port 8001 -target-port 8000
``` 

## 2️⃣ Start Messaging

- Type messages in either terminal and press Enter.
- Messages will appear in the other terminal.
- To exit, type: 
```bash
exit
```

This cleanly closes the connection and terminates the program.

---

## 📖 Command-line Flags

This cleanly closes the connection and terminates the program.

---

## 📖 Command-line Flags

| Flag          | Description                        | Default |
|:--------------|:-----------------------------------|:----------|
| `-local-port`  | Port to bind and listen for incoming messages | `8080`     |
| `-target-port` | Port to send outgoing messages             | `8081`     |

**Example:**
```bash
go run main.go -local-port 9000 -target-port 9001
``` 

## 📑 Example Output 

Binding to 127.0.0.1:8000
>>Type your message & press Enter (type 'exit' to quit)
>>Target: 127.0.0.1:8001
>> Hello there!
>> Received from 127.0.0.1:8001: Hi!
>> exit
Exiting...

## 📌 How It Works

- Opens a UDP socket bound to the given local port.

- Starts a receiver goroutine to continuously listen for incoming messages.

- Main thread reads user input and sends messages to the target port.

- On typing exit:

    Sends a dummy message ##exit## to itself to unblock the ReadFromUDP() call.

    Closes the receiver goroutine and exits gracefully.

