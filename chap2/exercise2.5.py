import math
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch



def f_sin(x):
    return torch.sin(x)

def f_tangent(f, x):
    h = 1e-5
    grad = (f(x+h) - f(x)) / h
    return grad

x_point=torch.arange(-5,5,0.1)
x_sin=f_sin(x_point)
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
axes.plot(x_point,x_sin,color='#00FF00')
y_tangent=f_tangent(f_sin,x_point)
axes.axhline(y=0, color='g', linewidth=1, linestyle='--')
axes.axvline(x=0, color='g', linewidth=1, linestyle='--')
axes.plot(x_point,y_tangent,color='#000000')
plt.show()
