import torch
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l
import matplotlib.pyplot as plt
print('2.5.1. A Simple Function')
x = torch.arange(4.0)
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
print(x.grad)  # The gradient is None by default
y = 2 * torch.dot(x, x)
print(y)
y.backward()
print(x.grad)
print("验证结果",x.grad==4*x)
x.grad.zero_()  # Reset the gradient,因为梯度的值会缓存
y = x.sum()
y.backward()
print(x.grad)#因为y=x1+x2+x3+x4
print('2.5.2. Backward for Non-Scalar Variables')
print("x=",x)
x.grad.zero_()
y = x * x
print("y=",y)
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
print(x.grad)
print('2.5.3. Detaching Computation')
x.grad.zero_()
y = x * x
u = y.detach()
u.requires_grad_(True)
z = u * x
print("x=",x)
print("u=",u)
print("z=",z)
z.sum().backward()#对z的组成中requires_grad=true的变量求导
print(x.grad == u)
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)
print("")
print('2.5.4. Gradients and Python Control Flow')
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(5,), requires_grad=True)
d = f(a)
d.sum().backward()
#print(a.grad==d/a)






