import torch
x = torch.tensor([[1,2], [3,4]])  # 形状 (2, 2)
x_repeat = x.repeat(3, 2)         # 第 0 维重复 3 次，第 1 维重复 2 次
print(x_repeat.shape)  # 输出 (2×3, 2×2) = (6, 4)
print(x_repeat)
# 输出：
# tensor([[1,2,1,2],
#         [3,4,3,4],
#         [1,2,1,2],
#         [3,4,3,4],
#         [1,2,1,2],
#         [3,4,3,4]])