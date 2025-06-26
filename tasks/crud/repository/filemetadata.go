package repository

import (
	"context"
	"fmt"

	"github.com/centauri1219/filenest/tasks/crud/model"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

type FileMetadataRepo struct {
	MongoCollection *mongo.Collection //collection on which we point our operations

}

func (r *FileMetadataRepo) InsertFile(fil *model.FileMetadata) (interface{}, error) { //interface is the object id thatll be created by mongodb
	result, err := r.MongoCollection.InsertOne(context.Background(), fil)
	if err != nil {
		return nil, err
	}

	return result.InsertedID, nil
}

func (r *FileMetadataRepo) FindFilebyID(fileID int) (*model.FileMetadata, error) {
	var file model.FileMetadata // we have to covert bson to struct field model.filemetadata
	err := r.MongoCollection.FindOne(context.Background(),
		bson.D{{Key: "id", Value: fileID}}).Decode(&file)

	if err != nil {
		return nil, err
	}

	return &file, nil

}

func (r *FileMetadataRepo) FindAllFiles() ([]model.FileMetadata, error) {
	results, err := r.MongoCollection.Find(context.Background(), bson.D{})
	if err != nil {
		return nil, err
	}
	var files []model.FileMetadata
	err = results.All(context.Background(), &files) //decodes and stores in files slice
	if err != nil {
		return nil, fmt.Errorf("results decode error %s", err.Error())
	}

	return files, nil
}

func (r *FileMetadataRepo) UpdateFilebyID(fileID int, updatefile *model.FileMetadata) (int64, error) {
	result, err := r.MongoCollection.UpdateOne(context.Background(),
		bson.D{{Key: "id", Value: fileID}},
		bson.D{{Key: "$set", Value: updatefile}})

	if err != nil {
		return 0, err
	}

	return result.ModifiedCount, nil //modified count: if you update the same doc again and again itll return 0
}

func (r *FileMetadataRepo) DeleteFilebyID(fileID int) (int64, error) {
	result, err := r.MongoCollection.DeleteOne(context.Background(),
		bson.D{{Key: "id", Value: fileID}})
	if err != nil {
		return 0, err
	}

	return result.DeletedCount, nil
}

func (r *FileMetadataRepo) DeleteAll() (int64, error) {
	results, err := r.MongoCollection.DeleteMany(context.Background(),
		bson.D{{}})

	if err != nil {
		return 0, err
	}

	return results.DeletedCount, nil
}
