# print("Starting train.py...")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from src.model import SmallCNN

import wandb
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def train_model(cfg: DictConfig):
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg.train, resolve=True)
    )

    # Transforms
    transform= transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Data
    train_dataset= datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset= datasets.MNIST(root="data", train=False, download=True, transform=transform)

    loaders= {
        'train': DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True),
        'test': DataLoader(test_dataset, batch_size=1000, shuffle=True)
    }

    # Model, loss, optimizer
    model= SmallCNN().to(device)
    wandb.watch(model, log="all", log_freq=100)

    criterion= nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.scheduler_step_size, gamma=cfg.train.scheduler_gamma)

    # Print parameter summary
    summary(model, input_size=(1, 1, 28, 28))

    # Training loop
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_loss, correct= 0, 0

        for images, labels in loaders['train']:
            images, labels= images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs= model(images)
            loss= criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += (loss.item())*images.size(0)
            preds= outputs.argmax(dim=1)
            correct += (preds==labels).sum().item()

        train_loss /= len(loaders['train'].dataset)
        train_accuracy= (100.*correct)/len(loaders["train"].dataset)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "epoch": epoch
        })

        evaluate_model(model, loaders["test"], device, epoch)

        scheduler.step()

    wandb.finish()


def evaluate_model(model, loader, device, epoch):
    model.eval()
    test_loss, correct= 0, 0
    criterion= nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs= model(images)
            loss= criterion(outputs, labels)
            test_loss += (loss.item())*images.size(0)
            preds= outputs.argmax(dim=1)
            correct += (preds==labels).sum().item()

    test_loss /= len(loader.dataset)
    accuracy= (100.*correct)/len(loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%\n")

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "epoch": epoch
    })

# print("Training run complete inside Docker!")
