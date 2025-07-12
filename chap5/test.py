import torch
from torchviz import make_dot
# 定义输入x并启用梯度追踪
x = torch.tensor(2.0, requires_grad=True)
k=torch.tensor(3,dtype=torch.int8)
y = x *k
z = 3 * y

make_dot(z, params={'x': x}).render("computation_graph", format="png")