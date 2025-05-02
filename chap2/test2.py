import math

import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(X<Y)
A=X.numpy()
print(A)
B=torch.from_numpy(A)
print(B)


