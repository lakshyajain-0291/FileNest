from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, shuffle=True):
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    return DataLoader(train, batch_size=batch_size, shuffle=shuffle), \
           DataLoader(test, batch_size=512, shuffle=False)
