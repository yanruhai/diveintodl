import torch
import matplotlib.pyplot as plt

x = torch.arange(12,dtype=torch.float32)
print(x.numel())#x的数量
X = x.reshape(3, 4)
print(X)
print(X.shape)
print("-------------------")
print(torch.randn(3, 4))
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X[-1], X[1:3])
X[:2, :] = 12
print(X)
print(torch.exp(x))#逐元素操作
print("-------------------")
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)
print("-------------------")
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))
print("-------------------")
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print("-------------------")
before = id(Y)
Y = Y + X
print(id(Y) == before)
print("-------------------2.1.5")
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
print("-------------------")
before = id(X)
X += Y
print(id(X) == before)
