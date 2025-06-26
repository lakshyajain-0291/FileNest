package main

import (
	"context" //something like control signals that handle how much time a request should take or so
	"fmt"     //i/o in go
	"log"

	"go.mongodb.org/mongo-driver/mongo"         //official mongo driver for go
	"go.mongodb.org/mongo-driver/mongo/options" //helps in configuring the connection to mongodb
)

func db() *mongo.Client { //creates a new mongo client and returns it
	clientOptions := options.Client().ApplyURI("mongodb+srv://posttoabhinavk:iy4AdEP9Bv5CVhwa@cluster0.t0oc6mk.mongodb.net/") //you can change the URI according to your mongodb cluster

	client, err := mongo.Connect(context.TODO(), clientOptions) //connects my Go program to MongoDB instance, client here will hold the client object while error will check while trying to connect
	//context.TODO is like non-nil empty context when not sure what context to use
	if err != nil { //if there is any sort of error, connection closes and error is shown
		log.Fatal(err)
	}

	err = client.Ping(context.TODO(), nil) //just checking the connection to MongoDB is working or not, nil cuz we are not making any changes to URI or someting
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Connected to MongoDB") //printing message to confirm connection is done
	return client
}

var fileCollection *mongo.Collection //global variable to hold all files metadata in MongoDB

func init() { //similar to C, runs before main
	client := db()                                                   //calling the above db function
	fileCollection = client.Database("fileNest").Collection("files") //allows you to call files in filenest database using filecollecion var
}
