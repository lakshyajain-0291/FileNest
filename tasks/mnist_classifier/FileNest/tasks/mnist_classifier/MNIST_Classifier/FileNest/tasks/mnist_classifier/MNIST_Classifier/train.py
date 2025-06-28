import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
import wandb

def train_func(config):
    from models.small_mobilenet import SmallMobileNet
    from data.mnist import get_dataloaders

    wandb.init(project="mnist", config=config)
    
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

    for epoch in range(config["epochs"]):
        model.train()
        total, correct, loss_sum = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['epochs']}]", leave=True)
        
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

        model.eval()
        test_total, test_correct = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                test_correct += (preds.argmax(1) == y).sum().item()
                test_total += y.size(0)
        test_acc = test_correct / test_total

        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
        })

    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print("Training complete!\n")
    print("Test Accuracy:", test_acc)
    

    wandb.finish()



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    config = {
        "batch_size": cfg.dataset.batch_size,
        "lr": cfg.optimizer.lr,
        "epochs": cfg.trainer.epochs,
        "mixed_precision": cfg.trainer.mixed_precision,
    }

    train_func(config)


if __name__ == "__main__":
    main()
