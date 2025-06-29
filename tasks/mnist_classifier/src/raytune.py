import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ray import tune
from src.model import SmallCNN
import wandb
import hydra
from omegaconf import DictConfig


def train_tune(config, base_cfg=None):
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform= transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset= datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset= datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader= DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    test_loader= DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model= SmallCNN().to(device)
    criterion= nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(), lr=config["lr"])
    scheduler= torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=base_cfg.train.scheduler_step_size,
        gamma=base_cfg.train.scheduler_gamma
    )

    if base_cfg:
        wandb.init(
            project=base_cfg.wandb.project,
            name=f"ray-run-lr{config['lr']}-bs{config['batch_size']}",
            config=config
        )

    for epoch in range(1, base_cfg.train.epochs+1):
        model.train()
        correct=0

        for images, labels in train_loader:
            images, labels= images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs= model(images)
            loss= criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds= outputs.argmax(dim=1)
            correct+= (preds==labels).sum().item()

        train_accuracy= (100.*correct)/len(train_loader.dataset)

        # Evaluate
        model.eval()
        test_correct=0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels= images.to(device), labels.to(device)
                outputs= model(images)
                preds= outputs.argmax(dim=1)
                test_correct+= (preds==labels).sum().item()

        test_accuracy= (100.*test_correct)/len(test_loader.dataset)

        # ✅ Log to wandb with explicit step
        if base_cfg:
            wandb.log({
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "epoch": epoch
            }, step=epoch)

        scheduler.step()

        # ✅ Report to Ray Tune with epoch count too
        tune.report({
            "accuracy": test_accuracy,
            "epoch": epoch
        })

    if base_cfg:
        wandb.finish()



@hydra.main(config_path="../config", config_name="config", version_base=None)
def run_tune(cfg: DictConfig):
    param_space= {
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128])
    }

    tuner= tune.Tuner(
        tune.with_parameters(train_tune, base_cfg=cfg),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            num_samples=10
        )
    )

    results= tuner.fit()

    print("Best result:", results.get_best_result("accuracy", "max").metrics)
