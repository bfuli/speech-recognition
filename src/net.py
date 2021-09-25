# encoding=utf-8
# Author:fuli
# Date:2021/9/19
from torch import nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        cfg = [16, 32, 32, 64, 64, 128, 64]
        self.layers = self.make_layers(cfg, batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layers(x)
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.classifier(out)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(num_features=128 * 50)
        self.model = nn.Sequential(
            nn.Linear(128 * 50, 3000),
            nn.ReLU(),
            nn.Linear(3000, 500),
            nn.ReLU(),
            nn.Linear(500, 4)
        )

    def forward(self, input):
        out = self.flatten(input)
        out = self.bn(out)
        return self.model(out)
