import torch
from torch import nn

class MNISTModel(nn.Module): #defining a class MNIST from the base class of all Pytorch, nn.Module
    def __init__(self): 
        super().__init__() #calling the parent class constructor to initialize the model
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), #input channel is 1, output channel is 8, kernel size is 3x3, padding is 1 so that the output size is same as input size
            nn.BatchNorm2d(8), #batch normalization to normalize the output of the previous layer
            nn.ReLU(inplace=True), #introducing non-linearity using ReLU activation function, inplace=True means it will not create a new tensor, it will modify the existing one

            nn.Conv2d(8, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((3,3)), # adaptive average pooling to reduce the output size to 3x3, it will adaptively calculate the output size based on the input size
            nn.Dropout(0.2), # dropout layer to prevent overfitting, 20% of the neurons will be randomly set to zero during training
            nn.Flatten(), # flattening the output to convert it into a 1D tensor
            nn.Linear(24*3*3, 10) #connected layer with 216 input features and 10 output features as there are 10 classes
        )

    def forward(self, x):
        return self.net(x) #forward method to pass the input through the network and return the output