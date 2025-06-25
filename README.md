# FileNest Framework 🗂️🔍

**A Multi-Modal Intelligent Search Engine over Decentralized Data**

FileNest Framework represents a paradigm shift in information retrieval and content discovery, leveraging cutting-edge artificial intelligence and distributed computing technologies. The platform addresses the growing need for intelligent, scalable, and semantic search capabilities across diverse content types including text documents, research papers, images, and multimedia content.

## 🌟 Overview

FileNest combines the ease of intelligent discovery—like Google Drive Search or YouTube recommendations—with the resilience and freedom of P2P systems. It provides an intelligent search engine that runs across a decentralized, peer-based network, making content discovery as simple and powerful as searching Google — but without giving up ownership, privacy, or freedom.

## 🎯 Key Features

- **🧠 Multi-Modal Intelligence**: Advanced AI-powered content understanding for text, images, videos, PDFs and other file formats
- **🌐 Decentralized Architecture**: Peer-to-peer network system that creates fault tolerance and reduces central dependency  
- **📚 Academic Specialization**: Sophisticated research paper analysis capabilities with citation networks and semantic understanding
- **⚡ High Performance**: Sub-100ms search responses with 15-20 files/sec indexing throughput
- **🔒 Privacy-First**: No central servers, no data mining, no platform censorship

## 🏗️ Architecture

FileNest operates through a hierarchical distributed tagging system that combines intelligent embedding generation with peer-to-peer routing mechanisms:

### System Components

1. **Bootstrap Phase**: Generate and distribute Depth 1 Tagging Vectors (D1TVs) across the network
2. **Local Indexing**: Generate embeddings for files and route metadata through the hierarchical network
3. **Continuous Training**: Update tagging vectors as new content is added to maintain accuracy
4. **Query Phase**: Process queries through the distributed network with similarity-based pruning

## 🛠️ Tech Stack

### Backend & Network Infrastructure
- **Core Language**: Golang for local-first application logic and AI/ML integration
- **Network**: go-libp2p-kad-dht for discovery and network routing
- **Database**: SQLite/PostgreSQL for metadata, vector databases (Milvus/FAISS) for embeddings
- **Caching**: Redis for high-performance caching and session management

### AI/ML Components
- **NLP**: Hugging Face Transformers (BERT, RoBERTa, T5)
- **Embeddings**: Sentence-Transformers for semantic vector representations
- **Computer Vision**: OpenCV, PIL, and CLIP for image-text cross-modal understanding
- **Multimodal**: Unified embedding system for image, text, and video content

### Frontend
- **CLI**: Golang Binary Executables with Cobra CLI
- **Web Interface**: React.js for user interface development


## 📁 Project Structure

> **Disclaimer**: This is a tentative first draft that <i> may contain inaccuracies</i>. Content will be updated based on ongoing discussions and actual implementation progress.

```
FileNest/
├── .github/                # GitHub-specific configurations
│   ├── workflows/          # CI/CD pipeline workflows
│   ├── issue_templates/    # Issue templates for bug reports, feature requests, etc.
│   └── pull_request_template.md  # Template for PR descriptions
├── ai/                     # AI/ML models and scripts
├── backend/                # Backend implementation (Go)
├── frontend/               # Frontend (CLI & React.js code)
├── network/                # P2P networking code
├── shared/                 # Shared utilities
├── docs/                   # Documentation files
├── tests/                  # Test cases
├── scripts/                # Automation and utility scripts
├── examples/               # Example configurations and use cases
├── .gitignore              # Ignore unnecessary files
├── LICENSE                 # MIT License
├── README.md               # This file
└── CONTRIBUTING.md         # Contribution guidelines
```

## 🚀 Quick Start

> **Disclaimer**: This is a tentative first draft that <i> may contain inaccuracies</i>. Content will be updated based on ongoing discussions and actual implementation progress.

### Prerequisites

- Go 1.21 or higher
- Python 3.9 or higher
- Node.js 18 or higher (for frontend)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/lakshyajain-0291/FileNest.git
cd FileNest

# Set up backend dependencies
cd backend
go mod init filenest-backend
go mod tidy

# Set up AI/ML components
cd ../ai
pip install -r requirements.txt

# Set up frontend
cd ../frontend
npm install

# Run the system
./scripts/start.sh
```

## 🎯 Use Cases

- **🎓 Academic Collaboration**: Labs and researchers can share files and papers directly, searchable across institutions without cloud lock-in
- **📰 Censorship-Resistant Discovery**: Useful for journalists, archivists, or individuals seeking sensitive or restricted knowledge
- **🔗 Decentralized Knowledge Commons**: Like a smarter, distributed version of torrenting — but semantically searchable
- **🏢 Self-Hosted File Search**: Enabling secure internal discovery without exposing to external servers

## 📊 Performance Targets

| Metric | Current State | Target |
|--------|--------------|--------|
| Indexing Speed | 1 file/second | 15-20 files/second |
| Search Latency | 800-1200ms | Under 100ms |
| Concurrent Users | Single user | 1000+ users |
| Storage Overhead | 85% | Under 20% |

## 🤝 Contributing

We welcome contributions from developers, researchers, and anyone interested in decentralized technologies! Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

### Team Roles
- **AI/ML Research Engineers**: Work on semantic embeddings and multimodal content analysis
- **Backend & Network Developers**: Build core backend logic and P2P protocols
- **Frontend Developers**: Create CLI tools and web interfaces

## 📖 Documentation

- [Architecture Guide](docs/architecture.md) - Detailed system architecture
- [Contributing Guide](CONTRIBUTING.md) - Development environment setup and contribution guidelines
- [Project Roadmap](docs/roadmap.md) - Development milestones and timeline

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is part of **Summer RAID 2025** (Realm of Artificial Intelligence and Data) at IIT Jodhpur.

**Mentors**: Lakshya Jain, Aradhya Mahajan, Laksh Mendpara

---

**FileNest Framework** - Making decentralized data as discoverable as centralized systems, but with freedom, privacy, and ownership intact.
