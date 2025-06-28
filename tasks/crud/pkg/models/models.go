package models

import "time"

type FileMetadata struct {
    ID          int       `json:"id" db:"id"`
    FileName    string    `json:"filename" db:"filename"`
    FilePath    string    `json:"filepath" db:"filepath"`
    FileSize    int64     `json:"filesize" db:"filesize"`
    ContentType string    `json:"content_type" db:"content_type"`
    CreatedAt   time.Time `json:"created_at" db:"created_at"`
    UpdatedAt   time.Time `json:"updated_at" db:"updated_at"`
}