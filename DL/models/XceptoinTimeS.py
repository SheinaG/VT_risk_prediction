from DL.DL_utiles.base_packages import *
from .imports import *
from .layers import *


def Conv1d(ni, nf, kernel_size=None, ks=None, stride=1, padding='same', dilation=1, init='auto', bias_std=0.01,
           **kwargs):
    "conv1d layer with padding='same', 'causal', 'valid', or any integer (defaults to 'same')"
    assert not (kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if padding == 'same':
        if kernel_size % 2 == 1:
            conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=kernel_size // 2 * dilation, dilation=dilation,
                             **kwargs)
        else:
            conv = SameConv1d(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'causal':
        conv = CausalConv1d(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'valid':
        conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
    else:
        conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=padding, dilation=dilation, **kwargs)
    init_linear(conv, None, init=init, bias_std=bias_std)
    return conv


class XceptionModule(Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        ks = [ks // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        self.bottleneck = Conv1d(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList([SeparableConv1d(nf if bottleneck else ni, nf, k, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), Conv1d(ni, nf, 1, bias=False)])
        self.concat = Concat()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return x


class XceptionBlock(Module):
    def __init__(self, ni, nf, residual=True, **kwargs):
        self.residual = residual
        self.xception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for i in range(4):
            if self.residual and (i - 1) % 2 == 0: self.shortcut.append(
                BN1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out * 4 * 2, 1, act=None))
            n_out = nf * 2 ** i
            n_in = ni if i == 0 else n_out * 2
            self.xception.append(XceptionModule(n_in, n_out, **kwargs))
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for i in range(4):
            x = self.xception[i](x)
            if self.residual and (i + 1) % 2 == 0: res = x = self.act(self.add(x, self.shortcut[i // 2](res)))
        return x


class XceptionTime(Module):
    def __init__(self, c_in, c_out, nf=16, nb_filters=None, adaptive_size=50, **kwargs):
        nf = ifnone(nf, nb_filters)
        self.block = XceptionBlock(c_in, nf, **kwargs)
        self.head_nf = nf * 32
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size),
                                  ConvBlock(self.head_nf, self.head_nf // 2, 1),
                                  ConvBlock(self.head_nf // 2, self.head_nf // 4, 1),
                                  ConvBlock(self.head_nf // 4, c_out, 1),
                                  GAP1d(1))

    def forward(self, x):
        x = self.block(x)
        x = self.head(x)
        return x
