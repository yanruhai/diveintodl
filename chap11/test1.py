import torch

A = torch.tensor([1, 2, 3])
B = torch.tensor([10, 20, 30])

dists = A.reshape((-1, 1)) - B.reshape((1, -1))
print(dists)