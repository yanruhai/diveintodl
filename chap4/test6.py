import torch

# 创建示例张量
X = torch.tensor([[100.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 方法1：使用索引 [0]
X_max_1 = X.max(dim=1, keepdim=True)[0]
print("方法1 - X_max_1 的形状:", X_max_1.shape)
print("方法1 - X_max_1 的值:")
print(X_max_1)

# 方法2：元组解包
values, indices = X.max(dim=1, keepdim=True)
X_max_2 = values
print("\n方法2 - X_max_2 的形状:", X_max_2.shape)
print("方法2 - X_max_2 的值:")
print(X_max_2)

# 验证两种方法结果相同
print("\n两种方法结果是否相同:", torch.allclose(X_max_1, X_max_2))
