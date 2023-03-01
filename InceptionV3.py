import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import time

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        # premodel = './pretrained/inception_v3_google-1a9a5a14.pth'  # 这是模型保存的路径
        # self.inception = models.inception_v3(pretrained=False)
        # pre = torch.load(premodel)  # 进行加载
        # self.inception.load_state_dict(pre)
        for param in self.inception.parameters():
            param.requires_grad = True
    
        # Input size 
        first_conv_layer = [nn.Conv2d(2, 3, kernel_size=1, stride=1, bias=True),
                            nn.AdaptiveMaxPool2d(299),
                            self.inception.Conv2d_1a_3x3]
        self.inception.Conv2d_1a_3x3= nn.Sequential(*first_conv_layer)

        # Fit classifier
        self.inception.fc = nn.Sequential(
                                nn.Linear(2048, 35),
                                #nn.ReLU(inplace=True),
                                #nn.BatchNorm1d(2048),
                                #nn.Linear(2048, 1024),
                                #nn.ReLU(inplace=True),
                                #nn.BatchNorm1d(2048),
                                #nn.Linear(1024, 20)
                            )    
    


    def forward(self, x):
        if self.inception.training:
            z, _ = self.inception(x)
            # print('1111111111111')
        else:  
            z = self.inception(x)
        return z


    
if __name__ == '__main__':
    net = Net()
    t1=time.time()
    x = torch.randn(2, 2, 128, 128)
    out = net(x)
    t2 = time.time()
    print((t2 - t1) * 1000)
    # summary(net, input_size=(2, 128, 128))
