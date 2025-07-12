import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

#net(torch.tensor([[0,1.0],[1,1]]))
print(net[0].weight)