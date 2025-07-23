import torch
import torch.nn as nn

# 简单网络：仅 Flatten
class SimpleFlatten(nn.Module):
    def __init__(self):
        super(SimpleFlatten, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)

# 测试
X = torch.tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)
model = SimpleFlatten()
Y = model(X)  # (1, 4)
print(Y)  # tensor([[1., 2., 3., 4.]])
loss = Y.sum()
loss.backward()
print(X.grad.shape)  # (1, 1, 2, 2)
print(X.grad)  # tensor([[[[1., 1.], [1., 1.]]]])，梯度重塑