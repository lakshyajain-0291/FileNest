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

	log.Println("✅ MongoDB connected successfully")
	return client, nil
}

func GetRelayAddrFromMongo() ([]string, error) {
	godotenv.Load(".env")
	uri := os.Getenv("MONGO_URI")
	MongoClient, err := SetupMongo(uri)
	if(err != nil){
		return nil,err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	collection := MongoClient.Database("Addrs").Collection("relays")
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

func UpsertRelayAddr(client *mongo.Client, addr string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	collection := client.Database("Addrs").Collection("relays")

	filter := bson.M{"address": addr}
	update := bson.M{"$set": bson.M{"address": addr, "updatedAt": time.Now()}}

	opts := options.Update().SetUpsert(true)

	_, err := collection.UpdateOne(ctx, filter, update, opts)
	if err != nil {
		return fmt.Errorf("failed to upsert relay address: %w", err)
	}
	return nil
}