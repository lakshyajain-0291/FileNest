import torch.nn as nn

class SmallMobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        def dw_sep_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),

            dw_sep_conv(12, 24),
            nn.MaxPool2d(2),

            dw_sep_conv(24, 48),
            nn.MaxPool2d(2),

            dw_sep_conv(48, 96),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(96, 10)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
