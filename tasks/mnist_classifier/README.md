# 🖥️ MNIST Classifier 🚀

A PyTorch-based MNIST digit classifier achieving **<10k parameters** and **>98.5% test accuracy**.  
The project features:
- ✅ PyTorch CNN architecture
- ✅ WandB experiment tracking
- ✅ Ray Tune hyperparameter tuning
- ✅ Hydra config management
- ✅ Dockerized execution environment

---

## 📦 Project Structure
.
├── Dockerfile
├── requirements.txt
├── .gitignore
├── config/
│ ├── init.py
│ └── config.yaml
├── src/
│ ├── init.py
│ ├── model.py
│ ├── raytune.py
│ └── train.py
├── main.py
└── README.md

---

## 📋 Requirements

- Docker installed and running
- W&B account with your personal API key

---

## 🔧 Setup

### 1️⃣ Clone the repository

```bash
git clone <repo_url>
cd mnist_classifier
```

### Build the Docker image
```bash
docker build -t mnist-classifier .
```

## 🖥️ Running the Training Job
### 🚨 Set your W&B API key as an environment variable:

- Linux/macOS
```bash
export WANDB_API_KEY=your_api_key_here
```

- Windows (CMD)
```bash 
set WANDB_API_KEY=your_api_key_here
```

## Run the container 
```bash
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY mnist-classifier
```

## 📊 View Training Results

Your training metrics will be live-tracked in your W&B project:

👉 https://wandb.ai/

---

## 📦 Installing Dependencies (Optional Local Run)

If you wish to run the project locally without Docker:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## ⚠️ Important

- **Do NOT hardcode your W&B API key** into your code or this README.
- Always pass it as an environment variable like this:

```bash
export WANDB_API_KEY=your_api_key_here
```

## 📚 Tech Stack
- Python 3.11
- PyTorch
- torchvision
- wandb
- hydra-core
- ray tune
- docker


 
