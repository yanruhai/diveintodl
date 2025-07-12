import torch
import matplotlib.pyplot as plt

# 创建一个需要梯度的张量
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

# 尝试直接绘图（会报错）
plt.plot(x, y)  # 这行代码会报错

