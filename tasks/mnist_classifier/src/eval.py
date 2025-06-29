import torch
from torchinfo import summary #for printing the model summary layer by layer

from classifier import MNISTModel


def main():
    model = MNISTModel()
    summary(model, input_size=(1, 1, 28, 28)) #taking a sample in nchw format


if __name__ == "__main__":
    main()
