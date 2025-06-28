#this file defines the NN architecture
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):#preferrred depthwise seperable instead of MobileNetv2
    #helps reduce parameters by separating depthwise and pointwise convolutions
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightweightCNN(nn.Module):#main cnn model
    
    
    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        base_channels=21,
        dropout_rate=0.2,
        use_batch_norm=True,#centres activations around zero
        use_global_avg_pool=True,#replaces fully connected layer with global average pooling
    ):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_global_avg_pool = use_global_avg_pool
        
        # First conv block
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_batch_norm else nn.Identity()
        
        # Depthwise separable conv blocks
        self.conv2 = DepthwiseSeparableConv(base_channels, base_channels * 2)
        self.bn2 = nn.BatchNorm2d(base_channels * 2) if use_batch_norm else nn.Identity()
        
        self.conv3 = DepthwiseSeparableConv(base_channels * 2, base_channels * 4)
        self.bn3 = nn.BatchNorm2d(base_channels * 4) if use_batch_norm else nn.Identity()
        
        # Additional depthwise separable conv block
        self.conv_extra = DepthwiseSeparableConv(base_channels * 4, base_channels * 4)
        self.bn_extra = nn.BatchNorm2d(base_channels * 4) if use_batch_norm else nn.Identity()
        
        # Final conv to reduce channels
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 2, 1)
        self.bn4 = nn.BatchNorm2d(base_channels * 2) if use_batch_norm else nn.Identity()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classifier
        if use_global_avg_pool:
            self.classifier = nn.Linear(base_channels * 2, num_classes)
        else:
            # Calculate size after pooling
            self.classifier = nn.Linear(base_channels * 2 * 3 * 3, num_classes)
            
    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Depthwise conv blocks
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 7x7 -> 3x3
        
        # Extra block
        x = F.relu(self.bn_extra(self.conv_extra(x)))
        
        # Final conv
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling or flatten
        if self.use_global_avg_pool:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)
            
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(cfg):
    model = LightweightCNN(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
        dropout_rate=cfg.model.dropout_rate,
        use_batch_norm=cfg.model.use_batch_norm,
        use_global_avg_pool=cfg.model.use_global_avg_pool,
    )
    return model


if __name__ == "__main__":
    model = LightweightCNN()
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Detailed parameter count using torchinfo
    try:
        from torchinfo import summary
        summary(model, (1, 1, 28, 28))
    except ImportError:
        print("Install torchinfo for detailed model summary: pip install torchinfo")