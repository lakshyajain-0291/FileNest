package model

import (
	"time"
)

type FileMetadata struct {
	ID          int       `json:"id" bson:"id"`
	FileName    string    `json:"filename" bson:"filename"`
	FilePath    string    `json:"filepath" bson:"filepath"`
	FileSize    int64     `json:"filesize" bson:"filesize"`
	ContentType string    `json:"content_type" bson:"content_type"`
	CreatedAt   time.Time `json:"created_at" bson:"created_at"`
	UpdatedAt   time.Time `json:"updated_at" bson:"updated_at"`
}
