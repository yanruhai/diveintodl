import torch
a = torch.tensor([[1, 2], [3, 4],[5,6]])  # 2行2列
b = torch.tensor([[7, 8],[9,10],[11,12]])  # 2行2列

# dim=0堆叠（竖着堆）
stacked_0 = torch.stack([a, b], dim=0)
print(stacked_0)
# 结果形状为[2, 2, 2]，相当于在垂直方向堆叠

# dim=1堆叠（横着堆）
stacked_1 = torch.stack([a, b], dim=1)
# 结果形状为[2, 2, 2]，相当于在水平方向堆叠
print(stacked_1)