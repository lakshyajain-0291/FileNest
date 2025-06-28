from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist_train_loader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def mnist_test_loader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
