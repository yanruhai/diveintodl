import torch

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)
X = torch.cat((X, X + 1), 1)
print(X)