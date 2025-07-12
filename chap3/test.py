import torch
import torch.nn as nn

# 创建一个需要梯度的张量（如模型参数）
w = torch.tensor(2.0, requires_grad=True)
y = w ** 2
for ep in range(3):
# 定义函数并计算损失

    loss = y.mean()

# 反向传播
    loss.backward()

# 梯度保存在w.grad中
    print(w.grad)  # 输出: tensor(4.) （因为dy/dw = 2w，当w=2时梯度为4）