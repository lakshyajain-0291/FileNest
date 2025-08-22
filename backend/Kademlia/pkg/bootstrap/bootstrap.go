package libp2p

import (
	"context"
	"fmt"
	"time"
)

type BootstrapConfig struct {
	BootstrapAddr string        
	Timeout       time.Duration 
}

type NodeRole int

const (
	RoleBootstrap NodeRole = iota
	RoleRegular
)


func Bootstrap(ctx context.Context, cfg BootstrapConfig, dialFn func(string) error) (NodeRole, error) {
	if cfg.BootstrapAddr == "" { // if no bootstrap given
		fmt.Println("[BOOTSTRAP] No bootstrap address provided — becoming bootstrap node.")
		return RoleBootstrap, nil
	}
	
	fmt.Printf("[BOOTSTRAP] Attempting to connect to %s...\n", cfg.BootstrapAddr)

	err := dialFn(cfg.BootstrapAddr)
	if err != nil {
		fmt.Printf("[BOOTSTRAP] Failed to connect to bootstrap: %v\n", err)
		return RoleBootstrap, nil
	}

	fmt.Println("[BOOTSTRAP] Connected successfully — joining as a regular node.")
	return RoleRegular, nil
}

