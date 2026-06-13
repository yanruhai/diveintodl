import matplotlib.pyplot as plt
import numpy as np
import torch
from d2l import torch as d2l

eta = 0.6
def f_2d(len,xs):
    alpha=torch.tensor([2**(-i-1) for i in range(len)])
    return torch.dot(alpha,xs**2)

def momentum_2d(xs, vs):
    alpha = torch.tensor([2 ** (-i) for i in range(len)])#计算梯度的系数
    vs=beta*vs+alpha*xs#计算梯度同时更新vs
    etas = torch.full((len,), eta)
    xs=xs-etas*vs
    return xs,vs

def train_2d(trainer,len, steps=20, f_grad=None):
    """Optimize a 2D objective function with a customized trainer.

    Defined in :numref:`subsec_gd-learningrate`"""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    xs = torch.ones(len)
    etas=torch.full((len,),eta)
    results = []
    vs=torch.zeros(len)
    for i in range(steps):
        if f_grad:
            xs, vs = trainer(xs, vs, f_grad)
        else:
            xs, vs= trainer(xs, vs)
        results.append(torch.mean(xs))
    return results

len=7
xs=torch.ones(len)
print(f_2d(len,xs))
eta, beta = 0.6, 0.25
result=train_2d(momentum_2d,len,steps=1000)
print(result)
d2l.set_figsize()

ax=np.arange(0,5,0.005)
plt.plot(ax,result)
plt.show()
