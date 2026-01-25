import torch

valid=torch.tensor([[1, 3], [2, 4]])
print(valid.shape)
valid=valid.reshape(1,-1)
print(valid.shape)