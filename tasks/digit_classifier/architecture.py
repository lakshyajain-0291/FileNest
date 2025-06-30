import torch
import torch.nn as nn
import torch.nn.functional as F


# Implements a Bottleneck block inspired by MobileNetV2
class Bottleneck(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, exp_factor=1, stride=1) -> None:
        super().__init__()
        self.exp_factor = exp_factor
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        if (in_channels == None or out_channels == None):
            print("in_channels or out_channels not defined.")
            return

        self.ExpansionLayer = nn.Conv2d(in_channels, in_channels*exp_factor, 1)
        self.BN1 = nn.BatchNorm2d(in_channels*exp_factor)
        self.Depthwise = nn.Conv2d(
            in_channels*exp_factor, in_channels*exp_factor, 3, stride, 1)
        self.Projection = nn.Conv2d(in_channels*exp_factor, out_channels, 1)
        self.skip = False
        if (self.in_channels == self.out_channels and stride == 1):
            self.skip = True

    def forward(self, X):
        residual = X
        # print(f"Size of X is: {X.size()}")
        out = F.relu6(self.ExpansionLayer(X))
        # print(f"Size after Expansion is: {out.size()}")
        out = self.BN1(out)
        out = F.relu6(self.Depthwise(out))
        # print(f"Size after Depthwise is: {out.size()}")
        out = self.Projection(out)
        # print(f"Size after Proj is: {out.size()}")
        # print(f"Final: {out.size()}, Residual: {residual.size()}")
        if (self.skip):
            # print("Residual activated")
            out = torch.add(out, residual)
            return out
        # print("Residual not activated")
        return out


class Classifier(nn.Module):
    def __init__(self, f1: int = 8, f2: int = 4):
        super().__init__()
        self.f1 = f1
        self.f2 = f2

        self.CL1 = nn.Conv2d(1, f1, 3, 2)
        self.bneck1 = Bottleneck(f1, f2, 1, 1)
        self.bneck2 = Bottleneck(f2, f2, 2, 1)
        self.bneck3 = Bottleneck(f2, 8, 2, 2)
        self.block1 = nn.ModuleList([Bottleneck(8, 8, 2, 2) for _ in range(2)])
        self.bneck4 = Bottleneck(8, 12, 1, 2)
        self.fc = nn.Linear(12, 10)

    def forward(self, X):
        out = F.relu6(self.CL1(X))  # 32 maps, 13x13 img
        out = self.bneck1(out)
        out = self.bneck2(out)
        out = self.bneck3(out)

        for block in self.block1:
            out = block(out)

        out = self.bneck4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def predict_labels(self, X):
        y_proba = F.softmax(self.forward(X), dim=1)  # dim=1 is along row.
        y_hat = torch.argmax(y_proba, dim=1)
        return y_hat
