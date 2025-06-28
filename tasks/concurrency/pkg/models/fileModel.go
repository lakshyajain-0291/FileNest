package models

import (
	"fmt"
	"os"
	"time"

	"github.com/lib/pq"
)

type FileIndex struct {
    ID        int       `gorm:"primaryKey;autoIncrement"`
    Filename  string
    Filepath  string
    Embedding pq.Float64Array `gorm:"type:float8[]"`
    D1tvID    int
    IndexedAt time.Time `gorm:"autoCreateTime"`
}

// TableName overrides the table name used by GORM
func (FileIndex) TableName() string {
    return "file_index"
}

func GenerateFileIndices(dirPath string) []FileIndex{
	var fileIndices []FileIndex
	files, err := os.ReadDir(dirPath)
    if err != nil {
        panic(err)
    }
    for _, dirFile := range files {
		fname := dirFile.Name()
		fpath := fmt.Sprintf("%v/%v", dirPath,fname)
		fileIndices = append(fileIndices, FileIndex{ID: 0, Filename: fname, Filepath: fpath})
	}
	return fileIndices
}