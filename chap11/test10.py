import torch

num_steps=5
a=torch.arange(0,num_steps).reshape(-1,1)
b=torch.arange(0,8,2)+0.1
x=a/b
print(x)
