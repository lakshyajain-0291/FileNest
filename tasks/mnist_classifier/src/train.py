import hydra #for config managing
import torch 
import torchvision #for mnist dataset and transformations
import wandb #
from omegaconf import DictConfig #dictconfig is a class from omegaconf for structural config management
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast #for mixed precision training
from torch.utils.data import DataLoader
from torchvision import transforms #mainly for image preprocessing

from classifier import MNISTModel


@hydra.main(config_path=".", config_name="config.yaml") #hydra.main tells hydra to use the config file in the current directory with that name
def train(cfg: DictConfig):
    wandb.init(project="mnist_classifier", mode="offline") #new weights and biases run for project logging offline to prevent it syncing with my account on wanfb

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
# 
    train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]) #random rotation is for +/- 10 degrees for augmentation and normalize is according to mean and standard deviation of MNIST

    test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]) #no augmentation for test set since we want to test on unseen natural data and not create any artificial data
    
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=train_transform) #
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=test_transform) #mnist_dataset train test split, download = True will help us download the dataset

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True) #creating batches along with shuffle = true to mix data during training
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size)

    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler(enabled=cfg.train.mixed_precision) #handling scaling loss to prevent underflow in mixed precision training

    for epoch in range(cfg.train.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() #clearing previous gradients

            with autocast(enabled=cfg.train.mixed_precision): #autocast is for mixed precision training
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward() #scaling the loss and backpropagating
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval() #evaluation mode to disable dropout and batch normalization
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad(): #no gradients required during evaluation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1) #_ captures the max scores and pred gets the predicted class index
                test_correct += preds.eq(labels).sum().item() #comparing predicted labels with actual labels
                test_total += labels.size(0)
        test_loss /= test_total
        test_acc = test_correct / test_total

        wandb.log( #logging metrics for visualization in wandb
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        print(f"Epoch {epoch+1}: Train acc {train_acc:.4f}, Test acc {test_acc:.4f}")


if __name__ == "__main__": #for telling to run the code in this only if the script is run directly, not when imported
    train()
