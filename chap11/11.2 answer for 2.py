import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

from chap8.test4 import batch_size

d2l.use_svg_display()

# Define some kernels
def gaussian(x,sigma=torch.tensor(1)):
    return torch.exp(-x**2 / 2*sigma**2)

def boxcar(x):
    return torch.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x

def epanechikov(x):
    return torch.max(1 - torch.abs(x), torch.zeros_like(x))

fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))#所有子图共享y轴

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = torch.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    ax.plot(x.detach().numpy(), kernel(x).detach().numpy())
    ax.set_xlabel(name)

#d2l.plt.show()

def f(x):
    return 2 * torch.sin(x) + x  #模拟函数 yi=2sin(xi)+xi+埃普西隆


n = 40
x_train, _ = torch.sort(torch.rand(n) * 5)#torch.sort() 函数会返回一个包含两个元素的元组，即 (排序后的张量，原始元素的索引)
y_train = f(x_train) + torch.randn(n)#生成训练集
x_val = torch.arange(0, 5, 0.1)#验证集
y_val = f(x_val)#验证数据

def nadaraya_watson(x_train, y_train, x_val, kernel,sigma):
    dists = x_train.reshape((-1, 1)) - x_val.reshape((1, -1))#计算出x_train中每个点到x_val的每个点的距离

    #[[1, 1, 1],
    #[2, 2, 2],
    #[3, 3, 3]]  x_train会这样广播，x_val会沿x轴广播
    # Each column/row corresponds to each query/key
    k = kernel(dists,sigma).type(torch.float32)
    k=k.clone()
    k.fill_diagonal_(0.0)
    # Normalization over keys for each query
    attention_w = k / k.sum(0)#沿着第0维计算，保持第0维维度，其他维度展平
    y_hat = y_train@attention_w#矩阵乘法
    return y_hat, attention_w

def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            pcm = ax.imshow(attention_w.detach().numpy(), cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)


sigma=torch.tensor(1.0,requires_grad=True)
learning_rate=0.01
batch_size=8
while True:
    idx = torch.randperm(len(x_val))[:batch_size]#permute随机生成数列
    x_batch = x_val[idx]
    y_batch = y_val[idx]
    y_hat, attention_w = nadaraya_watson(x_train, y_train, x_batch, gaussian,sigma)
    loss=((y_batch-y_hat)**2).sum()
    if loss <0.05:
        print(f'loss={loss},sigma={sigma}')
        print(y_batch)
        print(y_hat)
        break
    else:
        print(f'loss={loss},sigma={sigma}')
    loss.backward()
    with torch.no_grad():#no_grad是一个类
        sigma-=learning_rate*sigma.grad
        sigma.grad.zero_()


plot(x_train, y_train, x_val, y_val, kernels, names)
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma):
    return (lambda x: torch.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
#plt.show()


