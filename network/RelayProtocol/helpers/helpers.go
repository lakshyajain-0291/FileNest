package helpers

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

var (
	MongoClient *mongo.Client
)

func SetupMongo(uri string) (*mongo.Client, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MongoDB: %w", err)
	}

	if err := client.Ping(ctx, nil); err != nil {
		return nil, fmt.Errorf("failed to ping MongoDB: %w", err)
	}

	MongoClient = client
	log.Println("✅ MongoDB connected successfully")
	return client, nil
}

func DisconnectMongo() {
	if MongoClient != nil {
		if err := MongoClient.Disconnect(context.Background()); err != nil {
			log.Println("⚠ Error disconnecting MongoDB:", err)
		} else {
			log.Println("🛑 MongoDB disconnected")
		}
	}
}

func GetRelayAddrFromMongo() ([]string, error) {
	godotenv.Load(".env")
	uri := os.Getenv("MONGO_URI")
	MongoClient, err := SetupMongo(uri)
	if(err != nil){

	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	db := MongoClient.Database("Addrs")
	collection := db.Collection("relays")
	
	_, err = collection.InsertOne(ctx, bson.M{"address": "bootstrap"})
	if err != nil {
		log.Fatal(err)
	}
	log.Println(collection.Name())
	cursor, err := collection.Find(ctx, bson.M{})
	if err != nil {
		return nil, fmt.Errorf("failed to fetch relay addresses: %w", err)
	}
	defer cursor.Close(ctx)

	var relayList []string
	for cursor.Next(ctx) {
		var doc struct {
			Address string `bson:"address"`
		}
		if err := cursor.Decode(&doc); err != nil {
			return nil, fmt.Errorf("failed to decode relay document: %w", err)
		}
		if strings.HasPrefix(doc.Address, "/") {
			relayList = append(relayList, strings.TrimSpace(doc.Address))
		}
	}

	return relayList,nil
}