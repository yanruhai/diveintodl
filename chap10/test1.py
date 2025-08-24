import torch
from torch import nn
from d2l import torch as d2l

def test(num_inputs, num_hiddens, sigma=0.01):
    init_weight = lambda shapea,shapeb: nn.Parameter(torch.randn(shapea,shapeb) * sigma)
    triple = lambda: (init_weight(num_inputs, num_hiddens),
                      init_weight(num_hiddens, num_hiddens),
                      nn.Parameter(torch.zeros(num_hiddens)))
    b_i = triple()
    return b_i

num_in=3
num_hi=3
print(test(num_in,num_hi))

