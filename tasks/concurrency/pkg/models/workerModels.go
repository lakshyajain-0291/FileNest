package models

import (
	"context"

	"gorm.io/gorm"
)

type FileJob struct {
	DB *gorm.DB
	D1TVS [][]float64
	FileIndex FileIndex
	Ctx context.Context
	Cancel context.CancelFunc
}