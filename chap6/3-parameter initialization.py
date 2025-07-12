import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
print(net(X).shape)

def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)#需要用专门的函数做初始化，防止计算图丢失等问题
        nn.init.zeros_(module.bias)

net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])

def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5#是用data变量，防止产生计算图

net.apply(my_init)
print(net[0].weight[:2])