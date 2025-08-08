import zmq
import time

# Example fake cluster data
fake_data = {
    "Clusters": [
        {
            "centroid": [0.1, 0.2, 0.3],
            "files": [
                {
                    "filename": "example.txt",
                    "metadata": {
                        "created_at": "2025-07-01T10:00:00Z",
                        "last_modified": "2025-07-05T12:00:00Z",
                        "file_size": 1.23
                    },
                    "embedding": [0.1, 0.2, 0.3]
                }
            ]
        }
    ]
}

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://127.0.0.1:5555")  # ML binds here

time.sleep(1)  # give Go time to connect

print("Sending fake cluster JSON to Go client...")
socket.send_json(fake_data)
print("Sent.")
