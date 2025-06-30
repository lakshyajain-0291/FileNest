import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
import hydra
from omegaconf import DictConfig, OmegaConf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initially i intended to build my own NN, but my architecture was able to yield 94% at best.
# Hence I took help from LLMs to construct this neural network. 
class OptimizedMiniMobileNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super(OptimizedMiniMobileNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        
        # First depthwise separable block
        self.dw_conv1 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, groups=12, bias=False)
        self.bn_dw1 = nn.BatchNorm2d(12)
        self.pw_conv1 = nn.Conv2d(12, 24, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw1 = nn.BatchNorm2d(24)
        
        # Second depthwise separable block with stride
        self.dw_conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, groups=24, bias=False)
        self.bn_dw2 = nn.BatchNorm2d(24)
        self.pw_conv2 = nn.Conv2d(24, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw2 = nn.BatchNorm2d(32)
        
        # Third depthwise separable block with stride
        self.dw_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False)
        self.bn_dw3 = nn.BatchNorm2d(32)
        self.pw_conv3 = nn.Conv2d(32, 24, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw3 = nn.BatchNorm2d(24)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(24, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # First depthwise separable block
        x = F.relu(self.bn_dw1(self.dw_conv1(x)))
        x = F.relu(self.bn_pw1(self.pw_conv1(x)))
        
        # Second depthwise separable block
        x = F.relu(self.bn_dw2(self.dw_conv2(x)))
        x = F.relu(self.bn_pw2(self.pw_conv2(x)))
        
        # Third depthwise separable block
        x = F.relu(self.bn_dw3(self.dw_conv3(x)))
        x = F.relu(self.bn_pw3(self.pw_conv3(x)))
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_transforms(cfg):
    """Create transforms based on config"""
    # Training transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation(cfg.augmentation.rotation_degrees),
        transforms.RandomAffine(degrees=0, translate=tuple(cfg.augmentation.translate)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(cfg.augmentation.normalize_mean), 
                           tuple(cfg.augmentation.normalize_std))
    ])
    
    # Test transforms without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(cfg.augmentation.normalize_mean), 
                           tuple(cfg.augmentation.normalize_std))
    ])
    
    return transform_train, transform_test

def train_mnist(config, cfg):
    learning_rate = config["lr"]
    batch_size = config["batch_size"]
    num_epochs = cfg.training.num_epochs

    # Create transforms using config
    transform_train, transform_test = create_transforms(cfg)

    train_dataset = datasets.MNIST(root=cfg.data.root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root=cfg.data.root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = OptimizedMiniMobileNet(
        num_classes=cfg.model.num_classes, 
        dropout=cfg.model.dropout
    ).to(device)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")
    assert total_params <= 10000, "Parameter count exceeds 10,000"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler using config
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg.ray_tune.scheduler.step_size, 
        gamma=cfg.ray_tune.scheduler.gamma
    )

    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum().item()
                num_samples += y.size(0)
        val_acc = num_correct / num_samples
        return val_acc

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Step the scheduler
        scheduler.step()
        
        # Use test_loader for validation accuracy
        val_acc = check_accuracy(test_loader, model)
        tune.report(
            metrics={
                "mean_accuracy": val_acc,
                "val_acc": val_acc,
                "epoch": epoch,
                "loss": loss.item()
            }
        )

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create parameter space using config
    param_space = {
        "lr": tune.loguniform(cfg.hyperparameter_search.lr_min, cfg.hyperparameter_search.lr_max),
        "batch_size": tune.choice(cfg.data.batch_sizes),
    }

    scheduler = ASHAScheduler(
        metric=cfg.hyperparameter_search.metric, 
        mode=cfg.hyperparameter_search.mode
    )

    # Create a wrapper function that includes cfg
    def train_with_config(config):
        return train_mnist(config, cfg)

    tuner = tune.Tuner(
        train_with_config,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=cfg.hyperparameter_search.num_samples
        ),
        run_config=tune.RunConfig(
            callbacks=[
                WandbLoggerCallback(
                    project=cfg.wandb.project,
                    log_config=True
                )
            ]
        )
    )

    results = tuner.fit()

    best_result = results.get_best_result(
        metric=cfg.hyperparameter_search.metric, 
        mode=cfg.hyperparameter_search.mode
    )
    df = best_result.metrics_dataframe
    print("Best config:", best_result.config)
    print("Accuracy: ", best_result.metrics)

if __name__ == "__main__":
    main()
