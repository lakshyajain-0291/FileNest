import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.features= nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier= nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 32 x 1 x 1
            nn.Flatten(),  # 32
            nn.Linear(32, 10)  # 320 params
        )

    def forward(self, x):
        x= self.features(x)
        x= self.classifier(x)
        return x
