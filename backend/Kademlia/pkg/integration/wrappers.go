package integration

import (
	"final/backend/pkg/kademlia"
)

func (ckh *ComprehensiveKademliaHandler) FindSimilar() ([]ckh.Node.storage.EmbeddingResult, error){
	return ckh.node.FindSimilar();
} 
