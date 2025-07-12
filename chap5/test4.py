import torch
import numpy as np

a = np.array([1], dtype=np.int8)
b = np.array([2.5], dtype=np.float32)
c = a * b  # 输出：array([2.], dtype=float32)（NumPy会自动提升类型）
print(c)