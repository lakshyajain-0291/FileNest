import torch
import torch.utils.data as data
import ray
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
ray.init(ignore_reinit_error=True)

mnist_trainset = datasets.MNIST(
    root=r'/home/xyphoes/Desktop/Projects/FileNest _Fork/FileNest_Fork/tasks/digit_classifier/data', train=True, download=False, transform=None)
mnist_testset = datasets.MNIST(
    root=r'/home/xyphoes/Desktop/Projects/FileNest _Fork/FileNest_Fork/tasks/digit_classifier/data', train=False, download=False, transform=None)

(xTrain, yTrain) = mnist_trainset.data.to(
    torch.float32), mnist_trainset.targets.to(torch.float32)
(xVal, yVal) = mnist_testset.data.to(
    torch.float32), mnist_testset.targets.to(torch.float32)
xTrain = xTrain / 255.0
xVal = xVal / 255.0
xTrain = xTrain.unsqueeze(1).to(device)  # shape: (N, 1, 28, 28)
xVal = xVal.unsqueeze(1).to(device)
yVal = yVal.long().to(device)
yTrain = yTrain.long().to(device)

train_ds = ray.put(data.TensorDataset(xTrain, yTrain))
val_ds = ray.put(data.TensorDataset(xVal, yVal))
