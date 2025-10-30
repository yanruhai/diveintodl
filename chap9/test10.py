import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

num_steps=32
tau=4
batch_size=16
T=1000
t=torch.tensor(list(range(T)))
t1=[t[i:i+num_steps+tau] for i in range(T-num_steps-tau)]
t1=torch.stack(t1,0)
t2=[t1[:,i:i+tau] for i in range(len(t1[0])-tau)]
t2=torch.stack(t2,1)
label=[t1[:,tau:]]
label=torch.stack(label,2)
print(t1)