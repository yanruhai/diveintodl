import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
X=torch.rand(4, 8)
Y = net(X)
print(Y.mean())

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


linear = MyLinear(5, 3)
print(linear.weight,linear.bias)

print(linear(torch.rand(2, 5)))


