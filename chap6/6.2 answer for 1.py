import torch
from torch import nn
from torch.nn import functional as F

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

nt=NestMLP()
nt(torch.tensor([1,0,1.1]))
for (name,param) in nt.named_parameters():
    print(name,param,param.data)
