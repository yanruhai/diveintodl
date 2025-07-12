import torch

t= torch.tensor([[1,2,3],[4,4,5],[1,1,2]])
print(t.argmax(dim=1))
