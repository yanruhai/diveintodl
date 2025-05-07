import numpy as np
import torch

x = torch.arange(4.0)
x.requires_grad_(True)
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
print(x.grad)  # The gradient is None by default
y = 2 * torch.dot(x,x)*torch.dot(x, x)
y.backward(retain_graph=True)
y.backward()
y.backward()
print(x.grad)