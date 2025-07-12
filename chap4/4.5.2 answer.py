import torch


x=torch.tensor(0,dtype=torch.int8)
while True:
    print(x)
    x=x+1
    if x==torch.inf:
        break
