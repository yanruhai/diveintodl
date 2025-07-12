import torch

# 定义变量 x 和 y，并启用梯度跟踪
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 定义函数 f(x, y) = x^2 * y + y^2
f = x**2 * y + y**2

# 计算一阶偏导数 df/dx
f.backward(create_graph=True, retain_graph=True)
grad_x = x.grad.clone()  # df/dx = 2xy
x.grad.zero_()
y.grad.zero_()

# 计算一阶偏导数 df/dy
f = x**2 * y + y**2  # 重新定义 f，因为计算图已释放
f.backward(create_graph=True, retain_graph=True)
grad_y = y.grad.clone()  # df/dy = x^2 + 2y
x.grad.zero_()
y.grad.zero_()

# 计算二阶偏导数 d^2f/dx^2
grad_x.backward(create_graph=True)  # 对 df/dx 求导
grad_xx = x.grad.clone()  # d^2f/dx^2 = 2y
x.grad.zero_()
y.grad.zero_()

# 计算二阶偏导数 d^2f/dy^2
grad_y.backward(create_graph=True)  # 对 df/dy 求导
grad_yy = y.grad.clone()  # d^2f/dy^2 = 2
x.grad.zero_()
y.grad.zero_()

# 计算交叉偏导数 d^2f/dxdy
grad_y.backward(create_graph=True)  # 对 df/dy 求导关于 x
grad_xy = x.grad.clone()  # d^2f/dxdy = 2x
x.grad.zero_()
y.grad.zero_()

# 计算交叉偏导数 d^2f/dydx
grad_x.backward(create_graph=True)  # 对 df/dx 求导关于 y
grad_yx = y.grad.clone()  # d^2f/dydx = 2x
x.grad.zero_()
y.grad.zero_()

# 输出结果
print("一阶偏导数 df/dx:", grad_x)
print("一阶偏导数 df/dy:", grad_y)
print("二阶偏导数 d^2f/dx^2:", grad_xx)
print("二阶偏导数 d^2f/dy^2:", grad_yy)
print("交叉偏导数 d^2f/dxdy:", grad_xy)
print("交叉偏导数 d^2f/dydx:", grad_yx)