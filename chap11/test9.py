import torch

expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)

# 测试
x = torch.tensor([1, 2, 3])
print("原始 x:", x)
print("原始 x id:", id(x))

result = expand_dims(x, 0)
views=x.unsqueeze(0)
print("\nexpand_dims 的结果:", result)
print("result id:", id(result))
print("x 和 result 相同吗?", x is result)  # False
print("result 是 x.unsqueeze(0) 吗?", result.equal(x.unsqueeze(0)))  # True

print(views.equal(result))