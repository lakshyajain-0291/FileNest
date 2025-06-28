import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo as info

import data  # assumed to contain DataLoader setup

# Select device: GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seed for reproducibility
seed = 42
torch.cuda.manual_seed(seed)  # sets seed for current GPU context only


class BottleneckLayer(nn.Module):
    def __init__(self, in_c, out_c, exp_f, stride=1):
        super().__init__()

        # Determine whether residual connection is possible
        self.use_res_connect = stride == 1 and in_c == out_c

        # Intermediate (expanded) channel size
        mid_c = in_c * exp_f

        # Bottleneck block: Expand → Depthwise → Project
        self.block = nn.Sequential(
            # 1x1 Convolution (Expansion phase)
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU6(inplace=True),
            # 3x3 Depthwise Convolution
            nn.Conv2d(
                mid_c,
                mid_c,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=mid_c,
                bias=False,
            ),
            nn.BatchNorm2d(mid_c),
            nn.ReLU6(inplace=True),
            # 1x1 Convolution (Projection phase)
            nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        # Apply residual connection if allowed
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class Model(nn.Module):
    def __init__(self, in_c=1, num_classes=10):
        super().__init__()

        # Initial standard convolution: 1×28×28 → 8×28×28
        self.conv1 = nn.Conv2d(in_c, 8, kernel_size=3, stride=1, padding=1)

        # Stack of Bottleneck Layers (MobileNetV2 style)
        self.block1 = BottleneckLayer(8, 16, exp_f=2, stride=2)  # 8×28×28 → 16×14×14
        self.block2 = BottleneckLayer(16, 16, exp_f=2, stride=1)  # 16×14×14 → 16×14×14
        self.block3 = BottleneckLayer(16, 32, exp_f=2, stride=2)  # 16×14×14 → 32×7×7
        self.block4 = BottleneckLayer(32, 32, exp_f=2, stride=1)  # 32×7×7 → 32×7×7

        # Global average pooling: 32×7×7 → 32×1×1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final linear classifier (flattened from 32)
        self.fc = nn.Linear(32, num_classes, bias=False)

    def forward(self, x):
        # Forward pass through each layer
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)  # Shape: (B, 32, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 32)
        return self.fc(x)  # Final logits (B, 10)


# Instantiate model and move to device
model = Model().to(device=device)

# Print model architecture and parameter summary
print(info.summary(model))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
