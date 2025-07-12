import torch

# 设置输入值并启用自动求导
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
x3 = torch.tensor(4.0, requires_grad=True)

# 前向传播
a = x1 + x2
y = a * x3

# 假设目标输出
y_target = torch.tensor(30.0)

# 计算损失函数（均方误差的一半）
loss = 0.5 * (y_target - y) ** 2

# 反向传播
loss.backward()

# 打印结果
print(f"前向传播结果: y = {y.item()}")
print(f"损失函数值: L = {loss.item()}")
print("\n反向传播梯度:")
print(f"dL/dx1 = {x1.grad.item()}")
print(f"dL/dx2 = {x2.grad.item()}")
print(f"dL/dx3 = {x3.grad.item()}")
loss.backward()