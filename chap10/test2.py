import os
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

sentences=['a','b']
array = torch.tensor([[4, 5, 6, 2, 0],  # 句子 1
                [6, 5, 7, 8, 2]])
valid_len = (array != 4).type(torch.int32).sum(1)#!=符号右侧做了广播,sum(1)得到y轴上的和
print(valid_len)

idx=slice(0,150)
print(idx)