# Import necessary libraries and modules
# For logging with Weights & Biases
from ray.air.integrations.wandb import WandbLoggerCallback
import torch
from ray import tune
import objective  # Custom objective function for training/evaluation
from omegaconf import OmegaConf
import hydra  # For managing experiment configs
from ray.tune import RunConfig

# Set random seed for reproducibility
torch.manual_seed(42)


def main(config):
    """
    Main function to run Ray Tune hyperparameter search.
    Sets up the tuner with resource allocation, search space, and logging.
    """
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective.objective),
            resources={"cpu": 24, "gpu": 1}  # Resource allocation per trial
        ),
        tune_config=tune.TuneConfig(
            metric="val_acc",
            mode="max",
            num_samples=config["num_trials"]),  # No. of tuning trials to run
        run_config=RunConfig(
            callbacks=[
                WandbLoggerCallback(
                    project="FileNest-MobileV2",
                    log_config=False)
            ]
        ),
        param_space=config
    )
    results = tuner.fit()  # Runs the tuner for given params on our model
    best_result = results.get_best_result("val_acc", "max")

    print(f"Best trial config: {best_result.config}\n\n\n\n\n\n\n")
    print(f"Best trial final validation metrics: {best_result.metrics} \n\n\n")


def convert_cfg(config):
    """
    Converts a configuration dictionary into a Ray Tune-compatible search space.
    Handles different types of hyperparameter distributions.
    """
    tune_config = {}
    for key, val in config.items():
        if (key == "f1" or key == "f2"):
            tune_config[key] = tune.choice(val)
        elif (key == "lr"):
            tune_config[key] = tune.loguniform(
                val[0], val[1])
        elif (key == "beta1" or key == "beta2"):
            tune_config[key] = tune.uniform(
                val[0], val[1])
        else:
            tune_config[key] = val
    return tune_config


@hydra.main(version_base=None, config_path="./configuration", config_name="config")
def load_cfg(cfg):
    """
    Hydra entry point.
    Loads configuration, converts it for Ray Tune, and starts the main process.
    """
    config = OmegaConf.to_container(
        cfg.config, resolve=True)  # Converts to a dict
    tune_config = convert_cfg(config)  # Prepare search space config
    print(tune_config)
    main(tune_config)


if __name__ == "__main__":
    load_cfg()
