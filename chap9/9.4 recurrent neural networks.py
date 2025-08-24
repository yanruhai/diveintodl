import torch
from d2l import torch as d2l


X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))

print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))