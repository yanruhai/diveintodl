import math

import torch

# 创建FP64张量
x1 = torch.tensor(3.0, dtype=torch.float64)
# 或等价写法
x2 = torch.tensor( 3.0, dtype=torch.double)

# 创建FP32张量（默认）
x3 = torch.tensor( 3.0, dtype=torch.float32)
# 或省略dtype（默认即为float32）
x4 = torch.tensor( 3.0)

# 创建BFLOAT16张量
x5 = torch.tensor( 3.0, dtype=torch.bfloat16)

# 创建FP16张量
x6 = torch.tensor( 3.0, dtype=torch.float16)
# 或等价写法
x7 = torch.tensor( 3.0, dtype=torch.half)



i=2
x1 = torch.tensor(0, dtype=torch.float32)
while True:
   t=x1
   x1=torch.exp(x1)
   print(x1)
   if x1==0:
       break
   x1=t-0.5
