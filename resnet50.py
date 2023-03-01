import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = True

        # Input size 2x128x128 -> 2x224x224
        first_conv_layer = [nn.Conv2d(2, 3, kernel_size=1, stride=1, bias=True),
                            nn.AdaptiveMaxPool2d(224),
                            self.resnet.conv1]
        self.resnet.conv1 = nn.Sequential(*first_conv_layer)

        # Fit classifier
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 35),

        )

    def forward(self, x):
        z = self.resnet(x)
        return z

if __name__ == '__main__':
    net = ResNet50()
    t1 = time.time()
    x = torch.randn(2, 2, 128, 128)
    out = net(x)
    t2 = time.time()
    print((t2 - t1) * 1000)
    # summary(net, input_size=(2, 128, 128))

