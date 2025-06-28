# 🧠 MNIST Classifier

A compact, configurable, and tunable digit classification model on the MNIST dataset using:

- 🐍 **PyTorch** (deep learning)
- ⚙️ **Hydra** (configuration management)
- 📊 **Weights & Biases (WandB)** (logging)
- 📈 **Ray AIR / Ray Tune** (hyperparameter tuning)
- 📦 `uv` (ultra-fast package manager via `pyproject.toml`)

---

## 🖼️ Overview

This project implements a MobileNet-inspired convolutional neural network to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), with support for:

- 🧪 Training & evaluation with AMP + LR scheduling
- 🔍 Tuning with Ray Tune or Ray AIR
- 🎛️ Modular config management via Hydra
- 📉 Live experiment tracking with Weights & Biases

---

## 📁 Directory Structure

```
tasks/mnist_classifier/
├── conf/
│ ├── config.yaml # Hydra root config
│ ├── dataset/mnist.yaml # DataLoader config
│ ├── model/small_mobilenet.yaml # Model config
│ ├── optimizer/adam.yaml # Optimizer config
│ └── trainer/default.yaml # Training params
├── data/ # Auto-downloaded MNIST data
├── models/
│ └── small_mobilenet.py # Custom lightweight CNN
├── train.py # Main training entry point
├── utils.py # Torchinfo summary, helper utils
├── README.md # ← you're here
├── pyproject.toml # uv dependency manager
```
