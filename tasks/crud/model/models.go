package model

import (
	"go.mongodb.org/mongo-driver/bson/primitive"
)

type FileMeta struct {
	ID          primitive.ObjectID `bson:"_id,omitempty" json:"id,omitempty"`
	FileName    string             `json:"filename"`
	Size        int64              `json:"size"`
	ContentType string             `json:"content_type"`
	UploadedAt  primitive.DateTime `json:"uploaded_at"`
}
