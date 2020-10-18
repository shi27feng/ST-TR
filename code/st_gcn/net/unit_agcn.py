# The based unit of graph convolutional networks.

import math
from abc import ABC

import torch
import torch.nn as nn
from torch.autograd import Variable

# from .net import conv_init

'''
This class implements Adaptive Graph Convolution. 
Function adapted from "Two-Stream Adaptive Graph Convolutional Networks 
for Skeleton Action Recognition" of 
Shi. et al. ("https://github.com/lshiwjx/2s-AGCN")
'''


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


class UnitGCN(nn.Module, ABC):
    def __init__(self,
                 in_channels,
                 out_channels,
                 adj,
                 coff_embedding=4,
                 num_subset=3,
                 use_local_bn=False,
                 mask_learning=False):
        super(UnitGCN, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.param_adj = nn.Parameter(adj)
        nn.init.constant_(self.param_adj, 1e-6)
        self.A = Variable(adj, requires_grad=False)
        print(type(self.A))
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x, label, name):
        N, C, T, V = x.size()
        adj = self.A.cuda(x.get_device())
        adj = adj + self.param_adj

        y = None
        for i in range(self.num_subset):
            adj_i = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            adj_j = self.conv_b[i](x).view(N, self.inter_c * T, V)
            adj_i = self.soft(torch.matmul(adj_i, adj_j) / adj_i.size(-1))  # N V V
            adj_i = adj_i + adj[i]
            adj_j = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(adj_j, adj_i).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)
