# Contributing to FileNest 🤝

Welcome to FileNest! We're excited that you want to contribute to our Multi-Modal Intelligent Search Engine over Decentralized Data. This guide will help you get started, whether you're a first-time contributor or an experienced developer.

## 🌟 First Time Contributing? You're Welcome Here!

Don't worry if you're new to open source! FileNest is designed to be a learning-friendly project. We believe everyone has something valuable to contribute, regardless of experience level.

### What You'll Learn
- **AI/ML**: Semantic embeddings, multimodal content analysis, and intelligent search
- **Distributed Systems**: P2P networking, consensus algorithms, and decentralized architectures  
- **Backend Development**: Go development, API design, and system architecture
- **Frontend Development**: React.js, CLI tools, and user experience design

## 🚀 Quick Start Guide

### Step 1: Set Up Your Development Environment

**Prerequisites:**
- Git (for version control)
- Go 1.21+ (for backend development)
- Python 3.9+ (for AI/ML components)
- Node.js 18+ (for frontend development)
- A code editor (VS Code recommended)

**Installation:**

```bash
# 1. Fork the repository on GitHub (click the "Fork" button)

# 2. Clone YOUR fork (replace YOUR_USERNAME)
git clone https://github.com/YOUR_USERNAME/FileNest.git
cd FileNest

# 3. Add the original repository as upstream
git remote add upstream https://github.com/lakshyajain-0291/FileNest.git

# 4. Install dependencies for each component
# Backend
cd backend && go mod tidy && cd ..

# AI/ML (if requirements.txt exists)
cd ai && pip install -r requirements.txt && cd ..

# Frontend (if package.json exists)  
cd frontend && npm install && cd ..
```

### Step 2: Find Something to Work On

**For Beginners:**
- Look for issues labeled `good first issue` or `beginner-friendly`
- Check out the [documentation improvements](docs/) - always a great starting point!
- Fix typos, improve comments, or add examples

**For Intermediate Contributors:**
- Issues labeled `help wanted` or `enhancement`
- Adding new features to existing components
- Writing tests for existing functionality

**For Advanced Contributors:**
- Complex features requiring multiple components
- Performance optimizations
- Architecture improvements

### Step 3: Create a Branch and Start Coding

```bash
# 1. Make sure you're on the main branch and up to date
git checkout main
git pull upstream main

# 2. Create a new branch for your feature/fix
git checkout -b your-feature-name

# Example branch names:
# git checkout -b add-pdf-support
# git checkout -b fix-search-latency
# git checkout -b improve-contributing-docs
```

### Step 4: Make Your Changes

**Best Practices:**
- Write clear, descriptive commit messages
- Keep changes focused and atomic
- Add tests if you're adding new functionality
- Update documentation if needed
- Follow the existing code style

**Commit Message Format:**
```
type: short description

Longer description if needed

Examples:
feat: add PDF content extraction support
fix: resolve search timeout issue
docs: improve contributing instructions for beginners
test: add unit tests for embedding generation
```

### Step 5: Test Your Changes

```bash
# Run tests for the component you changed
cd backend && go test ./... && cd ..
cd ai && python -m pytest && cd ..
cd frontend && npm test && cd ..

# Test the entire system if possible
./scripts/test.sh  # (if this script exists)
```

### Step 6: Submit a Pull Request

```bash
# 1. Push your changes to your fork
git push origin your-feature-name

# 2. Go to GitHub and create a Pull Request
# - Click "Compare & pull request" button
# - Fill out the PR template with details about your changes
# - Reference any related issues with "Fixes #123" or "Closes #456"
```

## 🎯 Contribution Areas

### 🧠 AI/ML Components (`ai/`)
**What you'll work on:**
- Semantic embedding pipelines
- Multimodal content analysis (text, images, videos)
- Natural language processing for search queries
- Vector similarity algorithms

**Good for:** Data scientists, ML engineers, Python developers

**Learning opportunities:** Production ML systems, vector databases, semantic search

### 🔧 Backend (`backend/`)
**What you'll work on:**
- Core Go application logic
- API endpoints and services
- Database integration
- System architecture

**Good for:** Backend developers, systems programmers, Go enthusiasts

**Learning opportunities:** Distributed systems, high-performance backends, Go programming

### 🌐 Network Layer (`network/`)
**What you'll work on:**
- P2P protocol implementation
- Distributed hash tables (DHT)
- Network routing and discovery
- Consensus mechanisms

**Good for:** Network programmers, distributed systems enthusiasts

**Learning opportunities:** P2P protocols, networking, decentralized systems

### 🖥️ Frontend (`frontend/`)
**What you'll work on:**
- React.js web interface
- CLI tools and commands
- User experience design
- Search result visualization

**Good for:** Frontend developers, UX designers, React developers

**Learning opportunities:** Modern web development, CLI design, search interfaces

### 📚 Documentation (`docs/`)
**What you'll work on:**
- Contributing guides and tutorials
- Architecture documentation
- API documentation
- Examples and use cases

**Good for:** Technical writers, beginners, anyone wanting to help others

**Learning opportunities:** Technical communication, project documentation

## 💡 Tips for Success

### For First-Time Contributors

1. **Start Small**: Look for documentation improvements or simple bug fixes
2. **Ask Questions**: Use GitHub issues or discussions - we're here to help!
3. **Read the Code**: Browse existing code to understand patterns and style
4. **Follow Examples**: Look at existing PRs to see how others structure their contributions

### Communication Guidelines

- **Be Respectful**: We're all learning and growing together
- **Be Patient**: Reviews take time, especially for complex changes
- **Be Descriptive**: Clearly explain what you're doing and why
- **Ask for Help**: Stuck? Tag a maintainer or ask in discussions

### Code Quality Standards

- **Testing**: Add tests for new features
- **Documentation**: Update docs for user-facing changes
- **Performance**: Consider the impact of your changes on system performance
- **Security**: Be mindful of security implications, especially in network code

## 🏷️ Issue Labels Explained

- `good first issue` - Perfect for newcomers
- `beginner-friendly` - Suitable for those new to the codebase
- `help wanted` - We'd love community help on this
- `bug` - Something isn't working correctly
- `enhancement` - New feature or improvement
- `documentation` - Documentation related
- `question` - Need clarification or discussion

## 🎉 Recognition

We appreciate all contributions! Contributors will be:
- Added to our contributors list
- Mentioned in release notes for significant contributions
- Invited to join our contributor community

## ❓ Need Help?

**Quick Questions:**
- Comment on the issue you're working on
- Start a [GitHub Discussion](https://github.com/lakshyajain-0291/FileNest/discussions)

**Mentorship:**
- Reach out to our mentors:
  - Lakshya Jain: [@lakshyajain-0291](https://github.com/lakshyajain-0291)
  - Aradhya Mahajan: [@Aradhya2708](https://github.com/Aradhya2708)
  - Laksh Mandpara: [@Laksh-Mendpara](https://github.com/Laksh-Mendpara)

**Bug Reports:**
- Use our [bug report template](.github/issue_templates/bug_report.md)

**Feature Requests:**
- Use our [feature request template](.github/issue_templates/feature_request.md)

## 📋 Development Workflow

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes
5. **Test** your changes
6. **Commit** with clear messages
7. **Push** to your fork
8. **Create** a Pull Request
9. **Respond** to review feedback
10. **Celebrate** when merged! 🎉

## 🔄 Staying Updated

```bash
# Keep your fork synchronized
git checkout main
git pull upstream main
git push origin main
```

## 🚫 What NOT to Do

- Don't work directly on the main branch
- Don't submit PRs without testing
- Don't ignore the PR template
- Don't take feedback personally - it's about the code, not you!
- Don't hesitate to ask for help when stuck

---

## Thank You! 🙏

Every contribution, no matter how small, makes FileNest better. Whether you fix a typo, add a feature, or help another contributor, you're making a difference in building the future of decentralized, intelligent search.

**Remember**: The best contribution is the one you actually make. Start small, learn as you go, and don't be afraid to make mistakes. We're all here to learn and build something amazing together!

Happy coding! 🚀
