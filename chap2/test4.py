import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    return x**3 - 1/x

# 定义x=point的切线
def f_tangent(f, x, point):
    h = 1e-4
    grad = (f(point+h) - f(point)) / h
    return grad*(x-point) + f(point)

# 绘制函数图像和切线
x = np.arange(0.1, 2.0, 0.01)
y = f(x)
y_tangent = f_tangent(f, x=x, point=1)
plt.plot(x,y, label='f(x)')
plt.plot(x,y_tangent, label='Tangent line at x=1')
plt.legend()
plt.title('Graph of f(x) and its tangent line at x=1')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()