import time

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

T =10
time = torch.arange(1, T + 1, dtype=torch.float32)
a=x = torch.sin(0.01 * time) + torch.randn(T) * 0.2
print(a)