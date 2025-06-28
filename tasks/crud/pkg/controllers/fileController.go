package controllers

import (
	"context"
	"crud-api/pkg/config"
	"crud-api/pkg/models"
	"crud-api/pkg/utils"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/v2/bson"
)

// GetFiles retrieves all file metadata documents from the database and returns them as JSON.
func GetFiles(c *gin.Context){
	ctx := context.TODO()
	var Files []models.FileMetadata
	client := config.InitDB()
	filesColl := config.FetchDB(client, "Files")
	// Find all documents in the Files collection using bson.D{} as filter
	cur, err := filesColl.Find(ctx, bson.D{})
	if(err != nil){
		utils.ThrowError(c, err, http.StatusBadRequest, "Error on Find() on filesColl in GetFiles")
		return
	}
	err = cur.All(ctx, &Files)
	if(err != nil){
		utils.ThrowError(c, err, http.StatusBadRequest, "Error in cur.All in GetFiles")
		return
	}
	c.JSON(http.StatusOK, gin.H{"All Files: ": Files})

}

// GetFileById retrieves a single file metadata document by its ID and returns it as JSON.
func GetFileById(c *gin.Context){
	id, err := strconv.Atoi(c.Param("id"))
	if(err != nil){
		utils.ThrowError(c, err, http.StatusBadRequest, "Error while casting ID to int")
		return	
	}
	ctx := context.TODO()
	var File models.FileMetadata

	client := config.InitDB()
	filesColl := config.FetchDB(client, "Files")

	filter := bson.D{{Key: "id", Value: id}}
	err = filesColl.FindOne(ctx, filter).Decode(&File)
	if(err != nil){
		utils.ThrowError(c, err, http.StatusBadRequest, "Error while decoding file into struct in GetFileById")
		return
	}
	c.JSON(http.StatusOK, gin.H{"File: ": File})
}

// CreateFile creates a new file metadata document in the database.
func CreateFile(c *gin.Context){
	ctx := context.TODO()
	var File models.FileMetadata

	err := c.ShouldBindJSON(&File)
	if(err != nil){
		utils.ThrowError(c, err, http.StatusBadRequest, "Error on binding JSON in CreateFile")
	}
	client := config.InitDB()
	filesColl := config.FetchDB(client, "Files")
	log.Println(File.CreatedAt)
	File.CreatedAt = time.Now()

	_, err = filesColl.InsertOne(ctx,File)
	if(err != nil){
	utils.ThrowError(c, err, http.StatusBadRequest, "Error while upserting file in CreateFile")
	return
	}
	c.JSON(http.StatusOK, gin.H{"File Upserted Successfully: ": File})
}

// UpdateFile updates an existing file metadata document by its ID with only the provided fields.
func UpdateFile(c *gin.Context){
	id, err := strconv.Atoi(c.Param("id"))
	if(err != nil){
		utils.ThrowError(c, err, http.StatusBadRequest, "Error while casting ID to int")
		return	
	}
	ctx := context.TODO()
	updateData := make(map[string]any) // any == interface{}
	if err := c.ShouldBindJSON(&updateData); err != nil {
		utils.ThrowError(c, err, http.StatusBadRequest, "Error on binding JSON in UpdateFile")
		return
	}
	client := config.InitDB()
	filesColl := config.FetchDB(client, "Files")
	filter := bson.D{{Key:"id" , Value: id}}

	updateData["updated_at"] = time.Now()

	// Use $set flag to update only the provided fields
	update := bson.D{{Key: "$set", Value: updateData}}

	res, err := filesColl.UpdateOne(ctx, filter,update)
	if(err != nil){
		utils.ThrowError(c, err, http.StatusBadRequest, "Error while Updating file.")
		return
	}
	if(res.MatchedCount == 0){
		utils.ThrowError(c, nil, http.StatusBadRequest, "No matching file found for updation.")
		return
	}

	c.JSON(http.StatusOK, gin.H{"File Updated Successfully: ": update})
}

// DeleteFile deletes a file metadata document by its ID.
func DeleteFile(c* gin.Context){
	id, err := strconv.Atoi(c.Param("id"))
	if(err != nil){
		utils.ThrowError(c, err, http.StatusBadRequest, "Error while casting ID to int")
		return	
	}
	ctx := context.TODO()

	client := config.InitDB()
	filesColl := config.FetchDB(client, "Files")
	filter := bson.D{{Key: "id", Value: id}}
	res, err := filesColl.DeleteOne(ctx, filter)
	if(err != nil){
		utils.ThrowError(c, err, http.StatusBadRequest, "Error in DeleteOne in DeleteFile")
		return
	} else if(res.DeletedCount == 0){
		utils.ThrowError(c, nil, http.StatusBadRequest, "No matching file found for deletion.")
		return
	}
	c.JSON(http.StatusOK, gin.H{"Status: ":"File Deleted Successfully"})
}
