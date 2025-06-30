import os
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from ray import tune

from data import mnist_test_loader, mnist_train_loader
from model import Model


def train_mobilenet_tune(config):
    # Instantiate the model and move to the specified device
    model = Model().to(config["device"])

    # Use DataParallel if multiple GPUs are available
    if config["device"] == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Select optimizer based on config
    optimizer = optim.Adam(model.parameters(),lr = config["lr"])

    criterion = nn.CrossEntropyLoss()

    # Restore from checkpoint if available (for Ray Tune)
    if tune.get_checkpoint():
        with tune.get_checkpoint().as_directory() as ckpt_dir:
            model_state, opt_state = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(opt_state)

    # Prepare data loaders
    train_loader = mnist_train_loader(batch_size=config["batch_size"])
    test_loader = mnist_test_loader(batch_size=config["batch_size"])

    # Initialize Weights & Biases logging
    wandb.init(
        project=config.get("wandb_project", "mnist-raytune"),
        config=config,
        mode=config.get("wandb_mode", "online"),
        reinit=True,
    )

    for epoch in range(config["max_num_epochs"]):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        # Training loop
        for xb, yb in train_loader:
            xb, yb = xb.to(config["device"]), yb.to(config["device"])
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total

        # Validation loop
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(config["device"]), yb.to(config["device"])
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total
        val_loss /= len(test_loader)

        # Collect metrics for logging and Ray Tune reporting
        metrics = {
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "train_loss": avg_loss,
            "test_acc": val_acc,
            "test_loss": val_loss,
            "accuracy": val_acc,
            "loss": val_loss,
        }

        # Log metrics to wandb
        wandb.log(metrics)

        # Save checkpoint and report metrics to Ray Tune
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
            torch.save((model.state_dict(), optimizer.state_dict()), ckpt_path)
            checkpoint = tune.Checkpoint.from_directory(tmpdir)
            tune.report(metrics, checkpoint=checkpoint)

    # Finish wandb run
    wandb.finish()
