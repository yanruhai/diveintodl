import torch.nn as nn

module = nn.LazyLinear(10)
print(type(module))  # 输出: <class 'torch.nn.modules.lazy.LazyLinear'>
print(type(module) == nn.Linear)  # 输出: False
print(isinstance(module, nn.Linear))  # 输出: True