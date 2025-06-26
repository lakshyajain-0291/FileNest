package main

import (
	"time" //helps in created at and updated at stuff

	"go.mongodb.org/mongo-driver/bson/primitive" //mongodb uses ID as primitive object ID
)

type FileMetadata struct { //defining the file metdaata structure
	ID          primitive.ObjectID `json:"id,omitempty" bson:"_id,omitempty"`
	FileName    string             `json:"filename" db:"filename"`
	FilePath    string             `json:"filepath" db:"filepath"`
	FileSize    int64              `json:"filesize" db:"filesize"`
	ContentType string             `json:"content_type" db:"content_type"`
	CreatedAt   time.Time          `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time          `json:"updated_at" db:"updated_at"`
}
