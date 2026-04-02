import numpy as np
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x

def gd(eta, f_grad):#eta是学习率
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(1.1, f_grad)

def show_trace(results, f):
    #results是存储线段的节点列表，f是函数本身
    n = max(abs(min(results)), abs(max(results)))#找到最大的x坐标
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])
    plt.show()

show_trace(results, f)

c = torch.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * torch.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return torch.cos(c * x) - c * x * torch.sin(c * x)

show_trace(gd(2, f_grad), f)

def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    #trainer=gd_2d
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)#计算新的x1,x2坐标值
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')#画x1,x2连线,-o表示Orange
    #将类似results = [(-5, -2), (-4, -1.6), (-3.2, -1.28), (-2.56, -1.024)] 这种格式转成
    #[(-5, -4, -3.2, -2.56), (-2, -1.6, -1.28, -1.024)]
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                          torch.arange(-3.0, 1.0, 0.1), indexing='ij')#生成网格坐标
    #indexing='ij'（推荐）：第 1 个参数对应行维度，第 2 个参数对应列维度（符合矩阵索引习惯）；
    #indexing='xy'：第 1 个参数对应列维度，第 2 个参数对应行维度（符合笛卡尔坐标系习惯）。
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')#contour是等高线，以f(x1,x2)为标准画等高线
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    plt.show()

def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):#s1,s2暂时未用
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))

c = torch.tensor(0.5)

def f(x):  # Objective function
    return torch.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * torch.sinh(c * x)

def f_hess(x):  # Hessian of the objective function,一元函数的hessian就是二阶导
    return c**2 * torch.cosh(c * x)

def newton(eta=1):
    '''x -= eta * f_grad(x) / f_hess(x) 本质是阻尼牛顿法（Damped Newton Method），eta 就是 “阻尼系数”：
当 eta=1 时：等价于纯理论牛顿法，步长是理论最优值；
当 eta<1 时：缩小更新步长（比如 eta=0.5 就是只走理论步长的一半），避免步长过大导致的震荡；
当 eta>1 时：放大步长（慎用，容易发散）。'''
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)#12.3.9式结合梯度下降，x是一元的可以这样除法
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)

c = torch.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * torch.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return torch.cos(c * x) - c * x * torch.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * torch.sin(c * x) - x * c**2 * torch.cos(c * x)

show_trace(newton(0.5), f)
