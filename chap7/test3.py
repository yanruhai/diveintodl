import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # y = [1, 4, 9]
y.requires_grad_(True)
# 错误：in-place 操作修改了 y 的值
y += 10  # y 现在是 [11, 14, 19]，但 PyTorch 无法追踪原始 y 的值

z = y.sum()  # z = 11 + 14 + 19 = 44
z.backward()  # 反向传播时，PyTorch 不知道 y 原本的值，无法计算梯度