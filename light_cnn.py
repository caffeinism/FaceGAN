# https://github.com/AlfredXiangWu/LightCNN/blob/master/light_cnn.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

class network_9layers(nn.Module):
    def __init__(self):
        super(network_9layers, self).__init__()

        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.group2 = group(48, 96, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.group3 = group(96, 192, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.group41 = group(192, 128, 3, 1, 1)
        self.group42 = group(128, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.fc1 = mfm(8*8*128, 256, type=0)

    def forward(self, x):
        grayed = torch.mean(x, 1).unsqueeze(1)

        conv1 = self.conv1(grayed)
        pool1 = self.pool1(conv1)
        
        group2 = self.group2(pool1)
        pool2 = self.pool2(group2)

        group3 = self.group3(pool2)
        pool3 = self.pool3(group3)

        group41 = self.group41(pool3)
        group42 = self.group42(group41)
        pool4 = self.pool4(group42)

        flat = pool4.view(pool4.size(0), -1)
        fc1 = self.fc1(flat)

        return [x, pool1, pool2, pool3, pool4], fc1

def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs)
    return model