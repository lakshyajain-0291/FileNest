package config

import (
	"context"
	"log"
	"os"

	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

/*	InitDB initializes and returns a MongoDB client.
	It reads the MongoDB URI from the environment variable "MONGO_URI".
	If the URI is not set, the function logs a fatal error and exits.
 	It also pings the database to ensure the connection is established.*/
 func InitDB() *mongo.Client{
	uri := os.Getenv("MONGO_URI")
	if uri == ""{
		log.Fatal("Mongo URI not set in .env")
	}
	log.Println("Mongo URI Loaded")
	// Connect to MongoDB using the provided URI
	client, err := mongo.Connect(options.Client().ApplyURI(uri))
	log.Println("Mongo client loaded")
	if(err != nil){
		log.Fatalf("Error during connection to mongo: %s", err.Error())
	}
	// Ping the database to verify the connection
	var response bson.M
	err = client.Database("filesCluster").RunCommand(context.TODO(), bson.D{{Key:"ping", Value:1}}).Decode(&response)
	if(err != nil){
		log.Fatalf("Couldn't ping Mongo Cluster, Error: %s", err.Error())
	}
	log.Println("Successfully connected to mongo cluster!")
	return client
}

// FetchDB returns a collection handle from the "filesCluster" database.
func FetchDB(c *mongo.Client, collName string) *mongo.Collection{
	coll := c.Database("filesCluster").Collection(collName)
	return coll
}