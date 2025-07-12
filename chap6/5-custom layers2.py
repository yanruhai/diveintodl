import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
X = torch.tensor([1.0, 2, 3, 4, 5], requires_grad=True)
print(layer(X))
Y = X.sum()  # Y 是 X 的直接和
Y.backward()
print("X.grad:", X.grad)

net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())


