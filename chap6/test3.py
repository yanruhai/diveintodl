import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

torch.manual_seed(42)

def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight *= module.weight.abs() >= 5  # Modified line

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        shared = nn.LazyLinear(3)
        my_init(shared)
        self.net1 = shared
        self.net2 = shared
        self.relu = nn.ReLU()

    def forward(self, X):
        y1 = self.net1(X)
        y1_activated = self.relu(y1)
        y2 = self.net2(y1_activated)
        return y2.sum()

# Example usage
m = MLP()
X = torch.ones(3, 3, requires_grad=True)
t = m(X)
t.backward()
print("t:", t)
print("W.grad:\n", m.net1.weight.grad)
print("b.grad:\n", m.net1.bias.grad)
print("X.grad:\n", X.grad)