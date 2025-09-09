import torch
import torch
print(torch.__version__)  # 应显示 2.7.1+cu128
print(torch.cuda.is_available())  # 应返回 True
print(torch.version.cuda)  # 应显示 12.8
print(torch.cuda.get_device_name(0))  # 显示你的 GPU 名称，例如 "NVIDIA GeForce RTX 3080"