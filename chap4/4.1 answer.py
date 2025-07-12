import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

X = torch.tensor([[100.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

def softmax(X):
    (X_max,X_max_index)=X.max(dim=1,keepdim=True)
    X=X-X_max
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)#若原始维度：[m, n]，则求和后维度：[m, 1]
    return X_exp / partition  # The broadcasting mechanism is applied here

X=softmax(X)
print(X)