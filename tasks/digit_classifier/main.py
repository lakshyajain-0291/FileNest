from ray.air.integrations.wandb import WandbLoggerCallback
import wandb
import torch
import torch.nn as nn
from ray import tune
import objective
from omegaconf import OmegaConf
import hydra
from ray import air
from ray.tune import RunConfig
torch.manual_seed(42)


def main(config):
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective.objective),
            resources={"cpu": 24, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="val_acc",
            mode="max",
            num_samples=config["num_trials"]),
        run_config=RunConfig(
            callbacks=[
                WandbLoggerCallback(
                    project="FileNest-MobileV2",
                    log_config=False)
            ]
        ),
        param_space=config
    )
    results = tuner.fit()
    best_result = results.get_best_result("val_acc", "max")

    print(f"Best trial config: {best_result.config}\n\n\n\n\n\n\n")
    print(f"Best trial final validation metrics: {best_result.metrics}")


def convert_cfg(config):
    tune_config = {}
    for key, val in config.items():
        if (key == "f1" or key == "f2"):
            tune_config[key] = tune.choice(val)
        elif (key == "lr"):
            tune_config[key] = tune.loguniform(val[0], val[1])
        elif (key == "beta1" or key == "beta2"):
            tune_config[key] = tune.uniform(val[0], val[1])
        else:
            tune_config[key] = val
    # print(tune_config)
    return tune_config


@hydra.main(version_base=None, config_path="./configuration", config_name="config")
def load_cfg(cfg):
    config = OmegaConf.to_container(
        cfg.config, resolve=True)  # Convert to dict
    tune_config = convert_cfg(config)
    print(tune_config)
    main(tune_config)


if __name__ == "__main__":
    load_cfg()
