import torch

x=torch.arange(3,dtype=torch.float32)
print(x.shape)
A=torch.arange(6).reshape(3,2)
print(A)

A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
print(A, A + B)
print(A*B)#Hadamard product

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print(a + X, (a * X).shape)

print(A.shape, A.sum(axis=0))

print('2.3.7 Non-Reduction Sum')
print(A,A.shape)
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A, sum_A.shape)
print(A/sum_A)
print('2.3.8. Dot Products')
y = torch.ones(3, dtype = torch.float32)
print(x, y, torch.dot(x, y))
print(torch.sum(x * y))
print('2.3.9 Matrix–Vector Products')
print(A.shape, x.shape, torch.mv(A, x), A@x)
print('2.3.10. Matrix–Matrix Multiplication')
B = torch.ones(3, 4)
print(A,B,torch.mm(A, B), A@B)
print('2.3.11. Norms')
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
print(torch.abs(u).sum())
print(torch.norm(torch.ones((4, 9))))

