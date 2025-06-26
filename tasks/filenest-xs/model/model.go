package model

import (
	"time"

	"github.com/lib/pq"
)

type FileIndex struct {
	ID        uint `gorm:"primaryKey"`
	FileName  string
	FilePath  string
	Embedding pq.Float64Array `gorm:"type:float8[]"`
	D1TVID    int             `gorm:"column:d1tv_id"`
	IndexedAt time.Time       `gorm:"autoCreateTime"`
}

func (FileIndex) TableName() string {
	return "file_index"
}
