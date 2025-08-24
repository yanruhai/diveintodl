import torch
from torch import nn

def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)#这里的+形成新的元组(1,1,X.shape[0],X.shape[1])
    #为了适配conv2d类对于输入的要求
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])
X=torch.tensor([[1,2,3,4,5,6.0],[1,2,3,45,6,7],[1,2,3,45,6,7],[1,2,3,45,6,7],[1,2,3,45,6,7],[1,2,3,45,6,7]])
conv2d = nn.LazyConv2d(1, kernel_size=(3,5), padding=(0,1), stride=(3,4))
print(comp_conv2d(conv2d, X).shape)