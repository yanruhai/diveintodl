import torch

# 构造三维张量：shape [2, 3, 4]（dim0=2, dim1=3, dim2=4）
valid_lens_3d = torch.tensor([
    [[1,2,3,4], [5,6,7,8], [9,10,11,12]],  # dim0=0 对应的完整二维子张量
    [[13,14,15,16], [17,18,19,20], [21,22,23,24]]  # dim0=1 对应的完整二维子张量
])

print("原始三维张量：")
print(valid_lens_3d)
print("原始形状：", valid_lens_3d.shape)  # torch.Size([2, 3, 4])

# 核心操作：dim=0 上逐元素重复2次
result = torch.repeat_interleave(valid_lens_3d, repeats=2, dim=0)
print("\n" + "="*60)
print("dim=0 重复后三维张量：")
print(result)
print("重复后形状：", result.shape)  # torch.Size([4, 3, 4])