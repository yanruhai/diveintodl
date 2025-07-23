import torch
from torch import nn

def corr2d_multi_in_batch_optimized(X, K, stride=1):
    N, C_in, H_in, W_in = X.shape
    C_out, _, K_h, K_w = K.shape
    H_out = (H_in - K_h) // stride + 1
    W_out = (W_in - K_w) // stride + 1

    Y = torch.zeros((N, C_out, H_out, W_out))

    for n in range(N):  # 遍历批次
        for c_out in range(C_out):  # 遍历输出通道
            for i in range(0, H_in - K_h + 1, stride):
                for j in range(0, W_in - K_w + 1, stride):
                    for c_in in range(C_in):  # 遍历输入通道
                        X_slice = X[n, c_in, i:i + K_h, j:j + K_w]
                        Y[n, c_out, i // stride, j // stride] += torch.sum(X_slice * K[c_out, c_in])
                        #这里的*元素-wise 乘法

    return Y


# 测试
'''X = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
                  [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                   [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]])  # (2, 2, 3, 3)
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2, 2)
'''

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
X = torch.stack((X,X,X))
print('X shape',X.shape)
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
K = K.unsqueeze(0)
print('K shape',K.shape)
Y = corr2d_multi_in_batch_optimized(X, K, stride=1)
print("Input shape:", X.shape)  # torch.Size([2, 2, 3, 3])
print("Output shape:", Y.shape)  # torch.Size([2, 1, 2, 2])