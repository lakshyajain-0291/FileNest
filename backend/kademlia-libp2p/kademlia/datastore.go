package kademlia

import (
    "fmt"
    "path/filepath"
    "sync"
    "time"

    "gorm.io/driver/sqlite"
    "gorm.io/gorm"
    "gorm.io/gorm/logger"
    "github.com/libp2p/go-libp2p/core/peer"
)

type DataStore struct {
    db    *gorm.DB
    mutex sync.RWMutex
}

func NewDataStore(dataDir string) (*DataStore, error) {
    dbPath := filepath.Join(dataDir, "kademlia.db")
    
    // Open database with optimized settings
    db, err := gorm.Open(sqlite.Open(dbPath+"?_journal=WAL&_timeout=10000&_fk=true"), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Silent), // Reduce log noise
    })
    
    if err != nil {
        return nil, fmt.Errorf("failed to connect to database: %w", err)
    }

    // Configure SQLite for better performance
    sqlDB, err := db.DB()
    if err != nil {
        return nil, err
    }
    sqlDB.SetMaxOpenConns(1)      // SQLite only supports 1 connection
    sqlDB.SetMaxIdleConns(1)
    sqlDB.SetConnMaxLifetime(time.Hour)

    ds := &DataStore{db: db}
    
    // Auto-migrate tables
    if err := ds.migrate(); err != nil {
        return nil, err
    }

    return ds, nil
}

func (ds *DataStore) migrate() error {
    return ds.db.AutoMigrate(&KeyValue{}, &ContactRecord{}, &NodeStats{})
}

// Key-Value Operations
func (ds *DataStore) Put(key string, value []byte, ttl time.Duration) error {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()

    kv := KeyValue{
        Key:       key,
        Value:     value,
        CreatedAt: time.Now(),
    }

    if ttl > 0 {
        expires := time.Now().Add(ttl)
        kv.ExpiresAt = &expires
    }

    return ds.db.Save(&kv).Error
}

func (ds *DataStore) Get(key string) ([]byte, error) {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()

    var kv KeyValue
    err := ds.db.First(&kv, "key = ?", key).Error
    if err != nil {
        return nil, err
    }

    // Check if expired
    if kv.ExpiresAt != nil && time.Now().After(*kv.ExpiresAt) {
        // Delete expired key in background
        go ds.db.Delete(&KeyValue{}, "key = ?", key)
        return nil, fmt.Errorf("key expired")
    }

    return kv.Value, nil
}

func (ds *DataStore) Delete(key string) error {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()

    return ds.db.Delete(&KeyValue{}, "key = ?", key).Error
}

// Contact Operations
func (ds *DataStore) SaveContact(contact Contact) error {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()

    cr := ContactRecord{
        PeerID:   contact.ID.String(),
        NodeID:   fmt.Sprintf("%x", contact.NodeID),
        LastSeen: contact.LastSeen,
        IsActive: true,
    }
    cr.SetAddrs(contact.Addrs)

    return ds.db.Save(&cr).Error
}

func (ds *DataStore) LoadContacts() ([]Contact, error) {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()

    var records []ContactRecord
    err := ds.db.Where("is_active = ?", true).
              Order("last_seen DESC").
              Find(&records).Error
    
    if err != nil {
        return nil, err
    }

    var contacts []Contact
    for _, record := range records {
        peerID, err := peer.Decode(record.PeerID)
        if err != nil {
            continue
        }

        nodeID := make([]byte, KeySize)
        fmt.Sscanf(record.NodeID, "%x", &nodeID)

        contacts = append(contacts, Contact{
            ID:       peerID,
            NodeID:   nodeID,
            Addrs:    record.GetAddrs(),
            LastSeen: record.LastSeen,
        })
    }

    return contacts, nil
}

func (ds *DataStore) DeactivateContact(peerID peer.ID) error {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()

    return ds.db.Model(&ContactRecord{}).
              Where("peer_id = ?", peerID.String()).
              Update("is_active", false).Error
}

// Statistics Operations
func (ds *DataStore) UpdateStats(messagesSent, messagesRecv int64, peerCount, storedKeys int) error {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()

    stats := NodeStats{
        ID:           1, // Single stats record
        MessagesSent: messagesSent,
        MessagesRecv: messagesRecv,
        PeerCount:    peerCount,
        StoredKeys:   storedKeys,
    }

    return ds.db.Save(&stats).Error
}

func (ds *DataStore) GetStats() (*NodeStats, error) {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()

    var stats NodeStats
    err := ds.db.First(&stats, 1).Error
    return &stats, err
}

// Maintenance Operations
func (ds *DataStore) CleanupExpired() (int64, error) {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()

    // Clean expired key-value pairs
    result := ds.db.Where("expires_at IS NOT NULL AND expires_at < ?", time.Now()).
                  Delete(&KeyValue{})
    
    kvDeleted := result.RowsAffected

    // Deactivate old contacts (older than 2 hours)
    result = ds.db.Model(&ContactRecord{}).
                Where("last_seen < ? AND is_active = ?", time.Now().Add(-2*time.Hour), true).
                Update("is_active", false)
    
    contactsDeactivated := result.RowsAffected

    return kvDeleted + contactsDeactivated, result.Error
}

func (ds *DataStore) GetStorageInfo() (int64, int64, error) {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()

    var kvCount, contactCount int64
    
    ds.db.Model(&KeyValue{}).Count(&kvCount)
    ds.db.Model(&ContactRecord{}).Where("is_active = ?", true).Count(&contactCount)
    
    return kvCount, contactCount, nil
}

func (ds *DataStore) GetStoredKeys() []string {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()

    var keys []string
    ds.db.Model(&KeyValue{}).Pluck("key", &keys)
    return keys
}

func (ds *DataStore) Close() error {
    sqlDB, err := ds.db.DB()
    if err != nil {
        return err
    }
    return sqlDB.Close()
}
