import  torch

X = torch.ones(2, 3, 4, 4)
V=X.mean(dim=(0,2,3),keepdim=True)
print(V)
print(V.shape)