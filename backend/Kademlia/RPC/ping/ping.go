package ping

type PingRequest struct {
    SenderID   []byte `json:"sender_id"`
    SenderAddr string `json:"sender_addr"`
    Timestamp  int64  `json:"timestamp"`
}

type PingResponse struct {
    SenderID   []byte `json:"sender_id"`
    SenderAddr string `json:"sender_addr"`
    Timestamp  int64  `json:"timestamp"`
    Success    bool   `json:"success"`
}
