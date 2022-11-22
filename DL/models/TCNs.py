__all__ = ['TemporalBlock', 'TemporalConvNet', 'TCN']

from DL.DL_utiles.base_packages import *

from .layers import *
from .imports import *


# Cell
# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.
# Official TCN PyTorch implementation: https://github.com/locuslab/TCN


class TemporalBlock(Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, activation, dropout=0.):
        self.conv1 = weight_norm(nn.Conv1d(ni, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        if activation == 'RelU':
            act_func = nn.ReLU()
        if activation == 'leakyRelU':
            act_func = nn.LeakyReLU()
        self.relu1 = act_func
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(nf, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = act_func
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(ni, nf, 1) if ni != nf else None
        self.relu = act_func
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


def TemporalConvNet(c_in, layers, activation, ks=2, dropout=0.):
    temp_layers = []
    for i in range(len(layers)):
        dilation_size = 2 ** i
        ni = c_in if i == 0 else layers[i - 1]
        nf = layers[i]
        temp_layers += [TemporalBlock(ni, nf, ks, stride=1, dilation=dilation_size, padding=(ks - 1) * dilation_size,
                                      dropout=dropout, activation=activation)]
    return nn.Sequential(*temp_layers)


class TCN(Module):
    def __init__(self, c_in, c_out, layers=8 * [25], ks=7, conv_dropout=0., fc_dropout=0., activation='RelU'):
        self.tcn = TemporalConvNet(c_in, layers, activation, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1], c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)
