import math

import torch

c = torch.tensor(0.5)

def f(x1,x2):  # Objective function
    return  x1**3+math.exp(2*x2)

def f_grad(x1,x2):  # Gradient of the objective function
    return 3*x1**2,2*math.exp(2*x2)

def f_hess(x1,x2):  # Hessian of the objective function
    return 6*x1,4*math.exp(2*x2)

def newton(eta=1):
    x1,x2 = -1.0,-1.0
    results = []
    for i in range(10):
        hess=f_hess(x1,x2)
        grad1=f_grad(x1,x2)
        x1 -= eta * grad1[0] / abs(hess[0])
        x2 -= eta * grad1[1]/abs(hess[1])
        r=f(x1,x2)
        results.append(r)
        print(f'epoch:{i},x1:{x1},x2:{x2},f:{r}')
    print(f'epoch 10, x1:{x1},x2:{x2},f:{r}')
    return results

newton()

def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

