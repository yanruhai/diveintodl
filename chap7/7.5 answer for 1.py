import torch
from torch import nn

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def corr2d_avg(X, kshape):  #@save
    """Compute 2D cross-correlation."""
    ksize=kshape**2
    k = torch.full((kshape,kshape), fill_value=1/ksize)
    Y = torch.zeros((X.shape[0] - kshape + 1, X.shape[1] - kshape + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + kshape, j:j + kshape] * k).sum()
    return Y

X = torch.tensor([[1.0, 0.0, 0.0,0], [0.0, 1.0, 0.0,0], [0.0, 0.0, 1.0,0],[0,0,0,1]])


y=corr2d_avg(X,2)
print(y)
