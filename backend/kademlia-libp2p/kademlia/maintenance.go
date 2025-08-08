package kademlia

import (
    "context"
    "crypto/rand"
    "fmt"
    "time"
)

type MaintenanceManager struct {
    node          *Node
    cleanupInterval time.Duration
    refreshInterval time.Duration
    stopChan        chan struct{}
}

func NewMaintenanceManager(node *Node) *MaintenanceManager {
    return &MaintenanceManager{
        node:            node,
        cleanupInterval: 15 * time.Minute, // Cleanup every 15 minutes
        refreshInterval: 30 * time.Minute, // Refresh every 30 minutes
        stopChan:        make(chan struct{}),
    }
}

func (mm *MaintenanceManager) Start() {
    fmt.Println("Starting GORM-based maintenance manager")
    
    // Start cleanup routine
    go mm.cleanupLoop()
    
    // Start refresh routine
    go mm.refreshLoop()
    
    // Start statistics update routine
    go mm.statsLoop()
}

func (mm *MaintenanceManager) Stop() {
    close(mm.stopChan)
}

func (mm *MaintenanceManager) cleanupLoop() {
    ticker := time.NewTicker(mm.cleanupInterval)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            mm.performCleanup()
        case <-mm.stopChan:
            return
        }
    }
}

func (mm *MaintenanceManager) refreshLoop() {
    ticker := time.NewTicker(mm.refreshInterval)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            mm.performRefresh()
        case <-mm.stopChan:
            return
        }
    }
}

func (mm *MaintenanceManager) statsLoop() {
    ticker := time.NewTicker(5 * time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            mm.updateStatistics()
        case <-mm.stopChan:
            return
        }
    }
}

func (mm *MaintenanceManager) performCleanup() {
    fmt.Println("=== Database Cleanup ===")
    
    // Clean expired data from database
    deleted, err := mm.node.protocol.dataStore.CleanupExpired()
    if err != nil {
        fmt.Printf("Cleanup error: %v\n", err)
    } else {
        fmt.Printf("Cleaned up %d expired records\n", deleted)
    }
    
    // Clean stale routing table entries
    removed := mm.node.routing.Cleanup(2 * time.Hour)
    if removed > 0 {
        fmt.Printf("Removed %d stale contacts from routing table\n", removed)
    }
}

func (mm *MaintenanceManager) performRefresh() {
    fmt.Println("=== Routing Table Refresh ===")
    
    // Save current contacts to database
    mm.saveAllContacts()
    
    // Perform random lookups for discovery
    for i := 0; i < 3; i++ {
        randomID := generateRandomNodeID()
        _, cancel := context.WithTimeout(context.Background(), 30*time.Second)
        contacts, err := mm.node.FindNode(randomID)
        cancel()
        
        if err != nil {
            fmt.Printf("Random lookup %d failed: %v\n", i+1, err)
        } else {
            fmt.Printf("Random lookup %d discovered %d contacts\n", i+1, len(contacts))
        }
    }
}

func (mm *MaintenanceManager) saveAllContacts() {
    contacts := mm.node.routing.GetAllContacts()
    saved := 0
    
    for _, contact := range contacts {
        if err := mm.node.protocol.dataStore.SaveContact(contact); err == nil {
            saved++
        }
    }
    
    fmt.Printf("Saved %d contacts to database\n", saved)
}

func (mm *MaintenanceManager) updateStatistics() {
    sent, recv := mm.node.protocol.GetMessageStats()
    peerCount := mm.node.GetPeerCount()
    kvCount, _, _ := mm.node.protocol.dataStore.GetStorageInfo()
    
    err := mm.node.protocol.dataStore.UpdateStats(sent, recv, peerCount, int(kvCount))
    if err != nil {
        fmt.Printf("Failed to update statistics: %v\n", err)
    }
}

func generateRandomNodeID() []byte {
    nodeID := make([]byte, KeySize)
    rand.Read(nodeID)
    return nodeID
}
