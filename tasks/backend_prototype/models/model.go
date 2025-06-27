package models

import (
	"time"

	"github.com/lib/pq" //provides pq.Float64Array for []float64 type
)

type FileIndex struct {
	ID        uint `gorm:"primaryKey"` // telling gorm that this is the primary key
	Filename  string
	Filepath  string
	Embedding pq.Float64Array `gorm:"type:float8[]"` // using pq.Float64Array for []float64 type because gorm does not support []float64 directly
	D1TVID    int
	IndexedAt time.Time
}
