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

X = torch.tensor([[1.0, 0.0, 0.0,0], [0.0, 1.0, 0.0,0], [0.0, 0.0, 1.0,0],[0,0,0,1]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
K2=torch.tensor([[0,1.0],[1,0]])
K3=torch.tensor([[]])
print(corr2d(X,K))

print(corr2d(X,K2))
