"""
Xception
The network uses a modified version of Depthwise Seperable Convolution. It combines
ideas from MobileNetV1 like depthwise seperable conv and from InceptionV3, the order
of the layers like conv1x1 and then spatial kernels.
In modified Depthwise Seperable Convolution network, the order of operation is changed
by keeping Conv1x1 and then the spatial convolutional kernel. And the other difference
is the absence of Non-Linear activation function. And with inclusion of residual
connections impacts the performs of Xception widely.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SeparableConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.dwc = nn.Sequential(
            nn.Conv1d(input_channel, input_channel, kernel_size, stride, padding, dilation, groups=input_channel,
                      bias=bias),
            nn.Conv1d(input_channel, output_channel, 1, 1, 0, 1, 1, bias=bias)
        )

    def forward(self, X):
        return self.dwc(X)


class Block(nn.Module):
    def __init__(self, input_channel, out_channel, reps, strides=1, relu=True, grow_first=True, ks=9):
        super().__init__()
        if out_channel != input_channel or strides != 1:
            self.skipConnection = nn.Sequential(
                nn.Conv1d(input_channel, out_channel, 1, stride=strides, bias=False),
                nn.BatchNorm1d(out_channel)
            )
        else:
            self.skipConnection = None
        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = input_channel
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv(input_channel, out_channel, ks, stride=1, padding=int(ks / 2), bias=False))
            rep.append(nn.BatchNorm1d(out_channel))
            filters = out_channel

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv(filters, filters, ks, stride=1, padding=int(ks / 2), bias=False))
            rep.append(nn.BatchNorm1d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv(input_channel, out_channel, ks, stride=1, padding=int(ks / 2), bias=False))
            rep.append(nn.BatchNorm1d(out_channel))

        if not relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool1d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, input):
        X = self.rep(input)

        if self.skipConnection:
            skip = self.skipConnection(input)
        else:
            skip = input

        X += skip
        return X


class Xception1D(nn.Module):
    def __init__(self, input_channel, n_classes, ni=8, k=9, blocks=8, p_fc_drop=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.relu = nn.ReLU(inplace=True)

        self.initBlock = nn.Sequential(
            nn.Conv1d(input_channel, ni, k, 2, 1, bias=False),
            nn.BatchNorm1d(ni),
            nn.ReLU(inplace=True),

            nn.Conv1d(ni, ni * 2, kernel_size=k, padding=1, bias=False),
            nn.BatchNorm1d(ni * 2),
            nn.ReLU(inplace=True)
        )

        self.block1 = Block(ni * 2, ni * 4, 2, 2, relu=False, grow_first=True, ks=k)
        self.block2 = Block(ni * 4, ni * 8, 2, 2, relu=True, grow_first=True, ks=k)
        self.block3 = Block(ni * 8, ni * 16, 2, 2, relu=True, grow_first=True, ks=k)

        self.middle_blocks = nn.ModuleList()
        for i in range(blocks):
            self.middle_blocks.append(Block(ni * 16, ni * 16, 3, 1, relu=True, grow_first=True, ks=k))

        self.block12 = Block(ni * 16, ni * 32, 2, 2, relu=True, grow_first=False)

        self.conv3 = SeparableConv(ni * 32, ni * 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(ni * 64)

        # do relu here
        self.conv4 = SeparableConv(ni * 64, ni * 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm1d(ni * 128)

        self.fc = nn.Linear(ni * 128, self.n_classes)
        self.dropout = nn.Dropout(p=p_fc_drop)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.initBlock(x)
        x = self.block1(x)

        x = self.block2(x)
        x = self.block3(x)
        for block in self.middle_blocks:
            x = block(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
