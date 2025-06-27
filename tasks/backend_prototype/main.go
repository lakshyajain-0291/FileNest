package main

import (
	"backend_prototype/db" //custom package for database operations (InitDB)
	"context"
	"flag" //for command-line flags like -w, -dir, -timeout
	"log"
	"os"            //for file operations and environment variables (setting up postgres data source name (DSN))
	"os/signal"     //for ctrl+c handling
	"path/filepath" //for walking the directory structure
	"sync"          //for managing concurrent workers
	"time"
)

func main() {
	//all command line flags
	numWorkers := flag.Int("w", 5, "number of concurrent workers")                    //number of concurrent workers, default is 5
	dirPath := flag.String("dir", "./sample_texts", "directory to index")             //path to directory to index, default is "./sample_texts"
	timeout := flag.Duration("timeout", 5*time.Second, "per-file processing timeout") //timeout for processing each file, default is 5 seconds

	flag.Parse() //reading the flags the cient has set
	db.InitDB()  //initDB from database.go and gets the model fileindex from models/model.go

	// generating D1TV embeddings
	var d1tvs [][]float64 //multidimensional slice to hold D1TV embeddings, can help in automatically increasing the size of the slice as needed
	for i := 0; i < 10; i++ {
		d1tvs = append(d1tvs, generateEmbedding("D1TV")) //generateEmbedding is a function from worker.go that generates a D1TV embedding for each file
	}

	jobs := make(chan FileJob) //chanel to carry file jobs to workers

	ctx, stop := context.WithCancel(context.Background()) //ctx here is a context that is cancelled so as to stop workers at ctrl + c
	defer stop()                                          //using stop function as soon as main function exits, here stop is like a variable holding cancel function

	// handling ctrl + c to shutdown properly
	go func() { //startinga goroutine
		c := make(chan os.Signal, 1)   //made a channel to wait for an os signal and it can hold 1
		signal.Notify(c, os.Interrupt) //our program gets notified when we get an os.interrupt
		<-c                            //until someone writes into c channel
		log.Println("Interrupt received, shutting down...")
		stop()
	}()

	var wg sync.WaitGroup               //creating a pool of workers using sync.WaitGroup
	for i := 1; i <= *numWorkers; i++ { //from 1 to number of workers specified using the -w flag
		wg.Add(1)         //adding a worker
		go func(id int) { //here we are defining the function, calling it and launching it as a goroutine
			defer wg.Done()                        //later tell when the worker is done
			startWorker(id, jobs, d1tvs, *timeout) //startWorker is a function from worker.go that does the actual work of processing files
		}(i) //calling the function with id as i
	}

	// Walk the directory and send file paths to jobs channel
	err := filepath.Walk(*dirPath, func(path string, info os.FileInfo, err error) error { //starting to walk the directory specified by -dir flag, calling the function helps us recieve the string path, its metadata or any error it is facing
		select { //this select statement is used to handle the context cancellation
		case <-ctx.Done():
			return ctx.Err() //
		default: //for when context is not cancelled
			if err == nil && !info.IsDir() && filepath.Ext(path) == ".txt" { //ensures no error, the file is not soem kind of directory and filters for .txt file
				jobs <- FileJob{Path: path} //send the file path to jobs channel
			}
			return nil //return to show that no error is there
		}
	})
	if err != nil && err != context.Canceled { //if an error occurs and its not due to cancellation context
		log.Fatalf("Walk error: %v", err)
	}

	close(jobs) //close the jobs channel to tell that all files have been sent
	wg.Wait()   //waiting for workers to get processed
	log.Println("Indexing complete.")
}
