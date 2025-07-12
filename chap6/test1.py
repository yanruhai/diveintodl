import torch
from torch import nn

# 定义共享层
shared = nn.Linear(2, 2, bias=True)
shared.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
shared.bias.data = torch.tensor([0.5, 0.5])

# 模型：y1 = shared(x), y2 = shared(y1)
net = nn.Sequential(shared, shared)

# 输入
X = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)  # (2, 2)

# 前向传播
y1 = shared(X)  # y1 = w * x + b
y2 = shared(y1)  # y2 = w * y1 + b
loss = y2.sum()  # L = sum(y2)

# 反向传播
loss.backward()

# 输出梯度
print("Shared weight gradient:\n", shared.weight.grad)
print("Shared bias gradient:\n", shared.bias.grad)