import numpy as np

# 定义一个方阵
A = np.array([[2,1, 2],
              [1,2,2],
              [1,1,2]])

try:
    # 求矩阵的逆
    A_inv = np.linalg.inv(A)
    print("矩阵 A 的逆矩阵：")
    print(A_inv)
    print(np.dot(A_inv,A))
    print(np.dot(A_inv,[1,1,1]))
except np.linalg.LinAlgError:
    print("矩阵不可逆。")

