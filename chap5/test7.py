import torch

# 定义变量 x，并启用梯度跟踪
x = torch.tensor(2.0, requires_grad=True)

# 定义函数 f(x) = x^4
f = x ** 4

# 第一次 backward()，计算一阶导数 df/dx = 4x^3
f.backward(create_graph=True)
grad_1 = x.grad.clone()  # 保存一阶导数 4x^3 = 32（当 x=2）

# 清零梯度
x.grad.zero_()

# 对一阶导数 grad_1 再次求导，计算二阶导数 d^2f/dx^2
grad_1.backward()  # 对 grad_1 求导，得到 d^2f/dx^2 = 12x^2
grad_2 = x.grad  # 二阶导数 12x^2 = 48（当 x=2）

print("一阶导数:", grad_1)  # 应该输出 32.0
print("二阶导数:", grad_2)  # 应该输出 48.0