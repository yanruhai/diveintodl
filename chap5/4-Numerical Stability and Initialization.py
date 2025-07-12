import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

print('5.4.1.1. Vanishing Gradients')
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))#这里应该是y

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
#plt.show()
print('5.4.1.2. Exploding Gradients')
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    TM= torch.normal(0, 1, size=(4, 4))
    M = M @ TM
    print(M,TM)
print('after multiplying 100 matrices\n', M)