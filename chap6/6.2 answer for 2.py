import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        shared=nn.LazyLinear(3)
        self.net1=shared
        self.relu=nn.ReLU()
        self.net2=shared

    def forward(self,X):
        y1=self.net1(X)
        y1=self.relu(y1)
        y2=self.net2(y1)
        return y2.sum()

m=MLP()
X=torch.ones(3,3)
t= m(X)
print(m.net1.weight,m.net1.bias)
t.backward()
print(m.net1.weight.grad,m.net1.bias.grad)
print(X.grad)
print(t)



