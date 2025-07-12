import torch
from torch import nn
from torch.nn import functional as F

class Parallel_Module(nn.Module):
    def __init__(self,net1,net2):
        super().__init__()
        self.net1=net1
        self.net2=net2

    def forward(self,X):
        X1=self.net1(X)
        X2=self.net2(X)
        return torch.cat((X1,X2),dim=1)


pm=Parallel_Module(nn.Linear(10,12),nn.Linear(10,10))
X=torch.rand(2, 10)
print(pm(X))


