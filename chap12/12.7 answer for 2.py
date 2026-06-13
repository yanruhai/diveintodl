import math
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 4.2*x1-3.8*x2, -3.8*x1+4.2 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * (x1+x2) ** 2 + 2 * (x1-x2) ** 2

eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d,steps=140))
plt.show()