import hydra
from omegaconf import DictConfig
from ray import tune
from ray.tune import Tuner, TuneConfig, RunConfig

from data import mnist_train_loader, mnist_test_loader
from train import train_mobilenet_tune


# Helper to add train/test DataLoader to config (not used by Ray Tune, but can be handy)
def with_data_loader(cfg):
    cfg = cfg.copy()
    cfg["train_loader"] = mnist_train_loader(batch_size=cfg["batch_size"])
    cfg["test_loader"] = mnist_test_loader(batch_size=cfg["batch_size"])
    return cfg

search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),         # Learning rate sampled log-uniformly between 0.0001 and 0.01
    "batch_size": tune.choice([64, 128]),      # Batch size sampled from 64 or 128
    "max_num_epochs": 15,                      # Fixed number of epochs
    "num_trials": 2,                           # Number of Ray Tune trials (not part of search space per trial)
    "device": "cuda",  # Device setting
    "optim": "Adam",                           # Optimizer choice (fixed as Adam)
    "wandb_project": "mobilenetv2-mnist",      # WandB project name
    "wandb_mode": "online",                    # WandB logging mode
}
# Main entry point, managed by Hydra for config management
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Optionally add DataLoaders to config (not strictly needed for Ray Tune)
    cfg = with_data_loader(dict(cfg))

    # Set up Ray Tune Tuner for hyperparameter search
    tuner = Tuner(
        tune.with_resources(
            tune.with_parameters(train_mobilenet_tune),
            resources={"cpu": 2, "gpu": 1 if cfg["device"] == "cuda" else 0},
        ),
        tune_config=TuneConfig(
            metric="loss",         # Optimize for minimum validation loss
            mode="min",
            num_samples=cfg["num_trials"],  # Number of Ray Tune trials
        ),
        run_config=RunConfig(name="mobilenet_mnist"),
        param_space=search_space,         # Pass config as search space
    )

    # Run hyperparameter search
    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")

    # Print best trial configuration and final metrics
    print(f"\n Best trial config:\n{best_result.config}")
    print(f"Final val accuracy: {best_result.metrics['accuracy']:.4f}")
    print(f"Final val loss: {best_result.metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
