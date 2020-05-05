"""Backbone network - VGG16"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_channels,out,kernel,stride,padding,bias, n = 1):
    layers = nn.Sequential()
    in_channels = in_channels
    for i in range(n) :
        layers.add_module('conv_{}'.format(i),nn.Conv2d(in_channels, out, kernel_size = kernel, stride=stride, padding=padding, bias= bias))
        layers.add_module('bn_{}'.format(i),nn.BatchNorm2d(out,momentum=0.9, eps=1e-5))
        layers.add_module('relu_{}'.format(i),nn.LeakyReLU(0.1, inplace = True))
        in_channels = out
    return layers

class VGG16(nn.Module):

    def __init__(self,in_channels):
        super(VGG16,self).__init__()

        self.in_channels = in_channels
        self.Net()

    def Net(self):
        self.conv1 = conv_layer(in_channels = self.in_channels,out = 64,kernel = 3,stride = 1,padding = 1,bias = False, n = 2)
        self.conv2 = conv_layer(in_channels = 64,out = 128,kernel = 3,stride = 1,padding = 1,bias = False, n = 2)
        self.conv3 = conv_layer(in_channels = 128,out = 256,kernel = 3,stride = 1,padding = 1,bias = False, n = 3)
        self.conv4 = conv_layer(in_channels = 256,out = 512,kernel = 3,stride = 1,padding = 1,bias = False, n = 3)
        self.conv5 = conv_layer(in_channels = 512,out = 512,kernel = 3,stride = 1,padding = 1,bias = False, n = 3)

    def forward(self,x):
        out = F.max_pool2d(self.conv1(x),2)
        out = F.max_pool2d(self.conv2(out),2)
        out = F.max_pool2d(self.conv3(out),2)
        out = F.max_pool2d(self.conv4(out),2)
        out = self.conv5(out)
        return out


if __name__ == '__main__':
    a = VGG16(3)
    x = Variable(torch.Tensor(1,3,800,800))
    y = a(x)
    print (y.shape)
