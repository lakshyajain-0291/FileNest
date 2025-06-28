#loading and prepping MNIST
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(cfg):#normalise the data
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.dataset.normalize.mean, cfg.dataset.normalize.std),
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.dataset.normalize.mean, cfg.dataset.normalize.std),
    ])
    
    return train_transforms, test_transforms


def get_dataloaders(cfg):#loads mnist data and creates dataloaders based on the config
    train_transforms, test_transforms = get_transforms(cfg)
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root=cfg.dataset.data_dir,
        train=True,
        download=cfg.dataset.download,
        transform=train_transforms,
    )
    
    test_dataset = datasets.MNIST(
        root=cfg.dataset.data_dir,
        train=False,
        download=cfg.dataset.download,
        transform=test_transforms,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )
    
    return train_loader, test_loader


def get_dataset_info():
    #dataset info
    return {
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "train_size": 60000,
        "test_size": 10000,
    }