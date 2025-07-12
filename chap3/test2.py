import torch

x = torch.tensor(2.0, requires_grad=True)
y = (x ** 3)*torch.exp(x**0.5)

# 计算一阶导数
dydx = torch.autograd.grad(y, x, create_graph=True)[0]  # tensor(12.)

# 计算二阶导数
d2ydx2 = torch.autograd.grad(dydx, x, create_graph=True)[0]  # tensor(12.)


d3ydx3 = torch.autograd.grad(d2ydx2, x)[0]  # tensor(12.)
print(f"二阶导数: {d3ydx3}")