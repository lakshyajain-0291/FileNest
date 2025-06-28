import torch
import torch.nn as nn
from ray import tune
from torch.utils.data import DataLoader
import ray
import architecture
import load_data
import torchinfo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(config):
    batch_size = 64

    train_loader = DataLoader(
        ray.get(load_data.train_ds), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        ray.get(load_data.val_ds), batch_size, shuffle=True)

    # Init the model
    model = architecture.Classifier(config["f1"], config["f2"]).to(device)

    # Get total params
    total_params = torchinfo.summary(model, (64, 1, 28, 28)).trainable_params

    # Set up optimizer with config hyperparams
    optim = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(float(config["beta1"]), float(config["beta1"]))
    )
    loss_fn = nn.CrossEntropyLoss()

    # Training loop here
    for _ in range(config["max_num_epochs"]):
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

    model.eval()  # Set model to eval for validation set

    # Validation loop here
    with torch.no_grad():
        correct = 0
        val_loss = 0.
        for xb, yb in val_loader:
            logits = model(xb)
            val_loss += loss_fn(logits, yb)
            ypreds = model.predict_labels(xb)
            correct += torch.sum(ypreds == yb)

    val_acc = correct / len(load_data.xVal)
    val_loss = val_loss / len(load_data.xVal)

    # Prepare metrics for reporting to wandb and ray tuner
    metrics = {
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "total_params": total_params
    }

    # Saves the model and optim states
    # Create a new empty dir and pass the path for it in path.
    path = r"./model_saved"
    torch.save((model.state_dict(), optim.state_dict()),
               path + r'/checkpoint.pt')

    checkpoint = tune.Checkpoint.from_directory(path)
    tune.report(metrics, checkpoint=checkpoint)
