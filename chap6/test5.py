import torch
from torch import nn

x = torch.tensor([1, 2, 3])

# 方法1：使用 unsqueeze 增加维度，再进行矩阵乘法
x_col = x.unsqueeze(1)  # 形状变为 (3, 1) （列向量）
x_row = x.unsqueeze(0)  # 形状变为 (1, 3) （行向量）
result = torch.matmul(x_col, x_row)  # 结果为 (3, 3) 的矩阵

print("输入向量 x:", x)
print("列向量 x_col 形状:", x_col.shape)
print("行向量 x_row 形状:", x_row.shape)
print("相乘得到的矩阵:\n", result)
