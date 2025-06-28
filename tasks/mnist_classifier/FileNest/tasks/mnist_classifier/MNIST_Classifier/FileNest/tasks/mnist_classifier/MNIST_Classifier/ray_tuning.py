import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast, GradScaler 
from tqdm import tqdm

import ray
from ray import tune, air
from ray.air import session
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback

def train_func(config):
    from models.small_mobilenet import SmallMobileNet
    from data.mnist import get_dataloaders
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallMobileNet().to(device)

    train_loader, test_loader = get_dataloaders(
        batch_size=config["batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(device=device, enabled=config.get("mixed_precision", False))

    for epoch in range(1):
        model.train()
        total, correct, loss_sum = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{1}]", leave=False)
        
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=config.get("mixed_precision", False)):
                preds = model(x)
                loss = criterion(preds, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct / total) * 100:.2f}%",
            })

        train_acc = correct / total
        train_loss = loss_sum / len(train_loader)
        scheduler.step()

    # Evaluate
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    
    # Report back to Ray AIR
    session.report({
        "loss": train_loss,
        "accuracy": test_acc
    })

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    ray.init(
        include_dashboard=False, 
        num_cpus=7,
        num_gpus=1 if torch.cuda.is_available() else 0,
        ignore_reinit_error=True,
    )

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "epochs": cfg.trainer.epochs,
        "mixed_precision": cfg.trainer.mixed_precision,
    }

    scheduler = ASHAScheduler(
        max_t=cfg.trainer.epochs,
        grace_period=1,
        reduction_factor=2,
        metric="loss",    
        mode="min"    
    )

    tuner = tune.Tuner(
        train_func,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=10,
        ),
        run_config=RunConfig(
            name="mnist_tuning",
            callbacks=[WandbLoggerCallback(project="mnist", log_config=True)],
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best = results.get_best_result(metric="loss", mode="min")
    print("Best config:", best.config)
    print("Best test accuracy:", best.metrics["accuracy"])

if __name__ == "__main__":
    main()