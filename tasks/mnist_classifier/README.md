# MNIST Classifier

This project contains a machine learning pipeline for training and evaluating a classifier on the MNIST dataset. It is organized for reproducible experiments and supports configuration management and experiment tracking.

## Project Structure

```
mnist_classifier/
├── outputs/
│   └── <date>/<time>/
│       ├── .hydra/           # Hydra configuration snapshots
│       ├── data/             # MNIST data used in the run
│       └── wandb/            # Weights & Biases offline run logs
├── src/
│   ├── classifier.py         # Model definition
│   ├── config.yaml           # Default configuration file
│   ├── eval.py               # Evaluation script
│   ├── train.py              # Training script
│   └── __pycache__/          # Python bytecode cache
```

## Features

- Train and evaluate a classifier on the MNIST dataset
- Configuration management with Hydra
- Experiment tracking with Weights & Biases (wandb)
- Organized output directories for each run

## Getting Started

1. **Install dependencies:**
    ```sh
    uv pip install torch torchvision hydra-core wandb isort omegaconf black hydra-core torchaudio torchinfo
    ```

2. **Train the model:**
    ```sh
    python src/train.py
    ```

3. **Evaluate the model:**
    ```sh
    python src/eval.py
    ```

4. **Configuration:**
    - Modify `src/config.yaml` to change hyperparameters or paths.
    - Hydra will snapshot configs for each run in the corresponding `.hydra/` folder.

5. **Experiment Tracking:**
    - Each run logs outputs and configs in `outputs/<date>/<time>/`.
    - Weights & Biases logs are stored offline in the `wandb/` directory.

## Notes

- The MNIST data is stored under each run’s `data/MNIST/` directory.
- For reproducibility, all configs and logs are saved per run.
- Make sure to set up your wandb API key if you want to sync runs online.
- Also, tried using ray tuning for hyperparamters but due to its issues in Windows, I wasn't able to use it. 
- Also, used a normal CNN architecture instead of MobileNet V2 block because of its limitations in accuracy in simpler datasets like MNIST. 