import numpy as np
import time

# 创建两个大矩阵
n = 5000
A = np.random.rand(n, n).astype(np.float32)
B = np.random.rand(n, n).astype(np.float32)

# 计算乘法（这里会自动调用分块优化的 BLAS 库）
start = time.time()
C = A @ B
print(f"Time taken: {time.time() - start:.4f} seconds")