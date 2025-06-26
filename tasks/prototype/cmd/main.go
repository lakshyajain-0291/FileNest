package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/centauri1219/FileNest/tasks/prototype/internal/config"
	"github.com/centauri1219/FileNest/tasks/prototype/internal/database"
	"github.com/centauri1219/FileNest/tasks/prototype/internal/indexer"

	"github.com/spf13/cobra"
)

var (
	cfg      *config.Config     // used to access configuration settings throughout program
	db       *database.Database // connection to PostgreSQL database
	rootPath string             // stores the directory path to index
)

// basically sets up the command line interface (CLI) using Cobra to start indexing files
var rootCmd = &cobra.Command{ //main command for CLI
	Use:   "prototype",
	Short: "FileNest Local File Indexer Prototype",
	Long:  "A concurrent file indexer prototype that processes files and stores embeddings in Postgres",
}

var indexCmd = &cobra.Command{ // subcommand to index files in a directory
	Use:   "index [directory]",
	Short: "Index files in a directory",
	Args:  cobra.ExactArgs(1), //exactly one directory argument is required
	Run:   runIndex,           // tells cobra to call the runIndex function
}

func init() { //setup the CLI commands and flags
	rootCmd.AddCommand(indexCmd)                                                       // adds the index command as a subcommand of the root command
	indexCmd.Flags().StringVarP(&rootPath, "path", "p", "", "Directory path to index") //It allows users to optionally specify a directory path with -p or --path when running the index command.
}

func main() {
	// Load configuration settings like database URL, worker count, etc.
	cfg = config.LoadConfig()

	// Setup graceful shutdown by creating a context that can be cancelled
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle interrupt signals
	sigChan := make(chan os.Signal, 1)                      //to listen for OS interrupt like Ctrl+C os.signal is a buffer channel
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM) //registers sigchan to recieve SIGINT

	go func() { // go routine that waits for an interrupt signal
		sig := <-sigChan
		log.Printf("Received signal %v, shutting down gracefully...", sig)
		cancel() // cancel context
	}()

	// Initialize database
	var err error
	db, err = database.NewDatabase(ctx, cfg.DatabaseURL) // connects to the PostgreSQL database using the URL from the config
	if err != nil {
		log.Fatalf("Failed to initialize: %v", err)
	}
	defer db.Close()

	log.Println("FileNest Indexer Prototype initialized successfully")

	if err := rootCmd.ExecuteContext(ctx); err != nil { //starts the cobra CLI, passes context for shutdown
		log.Fatalf("Command execution failed: %v", err)
	}
}

func runIndex(cmd *cobra.Command, args []string) {
	dirPath := args[0] // directory path to index is passed as the first argument

	// Validate directory
	if _, err := os.Stat(dirPath); os.IsNotExist(err) { //checks if the directory exists
		log.Fatalf("Directory does not exist: %s", dirPath)
	}

	absPath, err := filepath.Abs(dirPath) // for consistency
	if err != nil {
		log.Fatalf("Failed to get absolute path: %v", err)
	}

	log.Printf("started indexing of directory: %s", absPath) //start of the indexing process
	log.Printf("worker count: %d", cfg.WorkerCount)
	log.Printf("process timeout: %d seconds", cfg.ProcessTimeout)
	log.Printf("embedding dimension: %d", cfg.EmbeddingDim) //prints out key configuration settings

	// Create indexer
	idx := indexer.NewIndexer(db, cfg.WorkerCount, cfg.ProcessTimeout, cfg.EmbeddingDim) //creates new indexer object

	// Start indexing
	if err := idx.IndexDirectory(cmd.Context(), absPath); err != nil { // calls Index Directory method to start processing files
		if err == context.Canceled {
			log.Println("Indexing cancelled") // error if the context was cancelled (like by Ctrl+C)
		} else {
			log.Fatalf("Indexing failed: %v", err) //any other error
		}
	}

	log.Println("Indexing completed successfully")
}
