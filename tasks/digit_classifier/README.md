# FileNest Digit Classifier

A PyTorch-based lightweight (<10k params) MNIST digit classifier with Ray Tune hyperparameter optimization, Hydra/OmegaConf configuration, and optional Weights & Biases (wandb) experiment tracking.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Training & Hyperparameter Tuning](#training--hyperparameter-tuning)
- [Logging & Experiment Tracking](#logging--experiment-tracking)
- [Developer Documentation](#developer-documentation)
  - [Key Modules](#key-modules)
  - [Adding New Models](#adding-new-models)
  - [Extending Data Loading](#extending-data-loading)
  - [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Structure

```
digit_classifier/
│
├── architecture.py         # Model and block definitions (Classifier, Bottleneck)
├── load_data.py            # Data loading and preprocessing
├── main.py                 # Entry point: config loading, Ray Tune setup, wandb integration
├── objective.py            # Objective function for Ray Tune (training, validation, reporting)
│
├── configuration/
│   └── config.yaml         # Hyperparameter search/configuration (Hydra/OmegaConf)
│
├── data/
│   └── MNIST/              # MNIST dataset (raw files)
│
└── model_saved/
    └── checkpoint.pt       # Saved model checkpoint
```

---

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Create a virtual environment** (recommended):

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies**:

    ```bash
    pip install torch torchvision ray[air] ray[tune] hydra-core omegaconf wandb
    ```

4. **Download MNIST data** (if not already present):

    The code expects MNIST data in `data/MNIST/raw/`.  
    If not present, set `download=True` in the `torchvision.datasets.MNIST` calls in your data loader.

---

## Configuration

All experiment and hyperparameter settings are managed via [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/).

Edit `configuration/config.yaml` to set search spaces and static parameters:

```yaml
config:
  f1: [8]
  f2: [6]
  lr: [1e-3, 1e-2]
  betas: [0.9, 0.999]
  max_num_epochs: 20
  num_trials: 3
```

- **Lists** are interpreted as search spaces (converted to `tune.choice`).
- **Scalars** are used as fixed values.

---

## Training & Hyperparameter Tuning

Run the main script to start Ray Tune hyperparameter optimization:

```bash
python main.py
```

- The script will:
  - Load the config via Hydra/OmegaConf.
  - Convert search spaces for Ray Tune.
  - Launch parallel trials for hyperparameter search.
  - Save the best model checkpoint to `model_saved/checkpoint.pt`.

**Results and logs** are stored in the `ray_results/` directory by default.

---

## Logging & Experiment Tracking

### Weights & Biases (wandb)

- To enable wandb logging, ensure you are logged in (`wandb login`) and the `WandbLoggerCallback` is included in `main.py`.
- Metrics and configs for each trial will be logged to your wandb project.

---

## Developer Documentation

### Key Modules

- **`architecture.py`**  
  Contains the `Classifier` and `Bottleneck` model classes.  
  Extend here to add new architectures.

- **`load_data.py`**  
  Handles loading and preprocessing of the MNIST dataset.  
  Modify to support new datasets or augmentations.

- **`objective.py`**  
  Defines the `objective(config)` function used by Ray Tune.  
  Handles training, validation, and metric reporting.

- **`main.py`**  
  Entry point. Loads config, sets up Ray Tune, and integrates wandb.

### Adding New Models

1. Define your model class in `architecture.py`.
2. Update the `objective` function in `objective.py` to use your new model.
3. Add any new hyperparameters to `config.yaml`.

### Extending Data Loading

- Modify or extend `load_data.py` to support new datasets or preprocessing steps.
- Ensure the data is loaded to the correct device and format.


## License

MIT License. See `LICENSE` file for details.

---