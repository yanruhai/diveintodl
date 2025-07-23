import torch
from torch import nn
from torch.nn import functional as F

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))

    def forward(self, X):
        X_flat=torch.flatten(X)
        x_col = X_flat.unsqueeze(1)  # 形状变为 (3, 1) （列向量）
        x_row = X_flat.unsqueeze(0)  # 形状变为 (1, 3) （行向量）
        X_new = torch.matmul(x_col, x_row)  # 结果为 (3, 3) 的矩阵
        return torch.matmul(torch.flatten(X_new),self.weight)