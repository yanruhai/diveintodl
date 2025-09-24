import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

def test(tt):
    print(tt)
x=torch.randn((10,20,30))
test(*x[:-1])