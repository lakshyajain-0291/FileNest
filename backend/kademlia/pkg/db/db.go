package db

import (
	"database/sql"
	"encoding/hex"

	_ "github.com/mattn/go-sqlite3"

	routingtable "kademlia/pkg/routingTable"
	"kademlia/pkg/types"
)

type DB struct {
    conn *sql.DB
}

func NewDB(path string) (*DB, error) {
    conn, err := sql.Open("sqlite3", path)
    if err != nil {
        return nil, err
    }

    schema := `
    CREATE TABLE IF NOT EXISTS peers (
        node_id TEXT NOT NULL,
        peer_id TEXT NOT NULL,
        bucket_index INTEGER NOT NULL,
        PRIMARY KEY (node_id, bucket_index)
    );`
    if _, err := conn.Exec(schema); err != nil {
        return nil, err
    }

    return &DB{conn: conn}, nil
}

func (db *DB) SavePeer(bucketIndex int, p types.PeerInfo) error {
    _, err := db.conn.Exec(`
        INSERT OR REPLACE INTO peers (node_id, peer_id, bucket_index)
        VALUES (?, ?, ?)`,
        hex.EncodeToString(p.NodeID),
        p.PeerID,
        bucketIndex,
    )
    return err
}

func (db *DB) LoadRoutingTable(rt *routingtable.RoutingTable) error {
    rows, err := db.conn.Query(`
        SELECT node_id, peer_id, bucket_index FROM peers
    `)
    if err != nil {
        return err
    }
    defer rows.Close()

    // Re-init buckets
    rt.Buckets = make([][]types.PeerInfo, len(rt.Buckets))

    for rows.Next() {
        var nodeIDHex, peerID string
        var bucketIndex int
        if err := rows.Scan(&nodeIDHex, &peerID, &bucketIndex); err != nil {
            return err
        }

        nodeID, _ := hex.DecodeString(nodeIDHex)
        peer := types.PeerInfo{
            NodeID: nodeID,
            PeerID: peerID,
        }

        // Expand bucket list if needed
        if bucketIndex >= len(rt.Buckets) {
            newBuckets := make([][]types.PeerInfo, bucketIndex+1)
            copy(newBuckets, rt.Buckets)
            rt.Buckets = newBuckets
        }

        rt.Buckets[bucketIndex] = append(rt.Buckets[bucketIndex], peer)
    }

    return nil
}
