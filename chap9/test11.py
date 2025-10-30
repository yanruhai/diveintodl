import torch
import numpy as np

# PyTorch示例
torch_tensor = torch.arange(30).reshape(5, 6)  # 创建一个5x6的张量用于演示
print("PyTorch原始张量:\n", torch_tensor)
print("原始形状:", torch_tensor.shape)  # 输出: torch.Size([5, 6])

# 隔一个取元素：提取偶数索引(0,2,4)和奇数索引(1,3,5)的元素
even_indices = torch_tensor[:, ::2]  # 取0,2,4列
odd_indices = torch_tensor[:, 1::2]  # 取1,3,5列

# 合并为(5,3,2)的三维张量，最后一维是[偶数索引元素, 奇数索引元素]
torch_result = torch.cat([even_indices, odd_indices], dim=1)
torch_result=torch_result.transpose(1, 2)
# 或者更直观的方式：
# torch_result = torch.stack([even_indices, odd_indices], dim=2).permute(0, 2, 1)

print("\nPyTorch处理后张量:\n", torch_result)
print("处理后形状:", torch_result.shape)  # 输出: torch.Size([5, 3, 2])






'''
# 使用NumPy的示例
# 创建一个形状为(5,6)的随机数组
np_array = np.random.randn(5, 6)
print("\nNumPy原始数组形状:", np_array.shape)  # 输出: (5, 6)

# 对第二维隔一个元素选取
np_result = np_array[:, ::2]
print("NumPy处理后数组形状:", np_result.shape)  # 输出: (5, 3)'''
