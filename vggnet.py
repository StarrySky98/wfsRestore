import torch
import torch.nn as nn
from torchsummary import summary
import time


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=32):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, out_channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'CBAM', 128, 128, 'CBAM', 256, 256, 256, 'CBAM', 512, 512, 512, 'CBAM', 512, 512, 512, 'CBAM'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(512 * 4 * 4, 512 * 4)
        self.bn_d1 = self._linear_layers()
        self.fc2 = nn.Linear(512 * 4, 128)
        # self.bn_d2 = self._linear_layers()
        self.fc3 = nn.Linear(128, 35)

    @staticmethod
    def _linear_layers():
        layers = list([])
        # layers.append(torch.nn.Dropout(0.5))
        layers.append(torch.nn.Tanh())
        return nn.Sequential(*layers)

    @staticmethod
    def _make_layers(structure):
        layers = []
        in_channels = 2
        for out_channels in structure:
            if out_channels == 'CBAM':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2),CBAM(in_channels)]
            else:
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                     padding=1, bias=False),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)]
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.bn_d1(out)
        out = self.fc2(out)
        # out = self.bn_d2(out)
        out = self.fc3(out)
        return out


if __name__ == '__main__':
    v16 = VGG('VGG16')
    print(v16)
    t1 = time.time()
    x = torch.randn(2, 2, 128, 128)
    out = v16(x)
    t2 = time.time()
    print((t2-t1)*1000)
    # summary(v16, input_size=(2, 128, 128))