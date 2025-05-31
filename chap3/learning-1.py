import math
import time

import d2l
import numpy as np
import torch
from matplotlib import pyplot as plt

print('3.1.2. Vectorization for Speed')
n = 10000
a = torch.ones(n)
b = torch.ones(n)
c = torch.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{time.time() - t:.5f} sec')
t = time.time()
d = a + b
print(f'{time.time() - t:.5f} sec')
print('3.1.3. The Normal Distribution and Squared Loss')
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)
# Use NumPy again for visualization
x = np.arange(-7, 7, 0.01)


# 生成数据
params = [(0, 1), (0, 2), (3, 1)]

# Generate x values
x = torch.linspace(-7, 7, 100)

# Compute the probability densities for each (mu, sigma) pair
y = [normal(x, mu, sigma).numpy() for mu, sigma in params]#numpy()是torch中将张量转成numpy数组的函数

# Plot the distributions
plt.figure(figsize=(4.5, 2.5))
for i, (mu, sigma) in enumerate(params):
    plt.plot(x.numpy(), y[i], label=f'mean {mu}, std {sigma}')
#Matplotlib 的颜色循环（Color Cycle)每次调用 plt.plot 绘制一条新曲线时，Matplotlib 会从颜色循环中按顺序选取下一个颜色。

# Add labels and legend
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.show()
