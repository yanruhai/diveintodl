import os
import pandas as pd
import torch


data_file = os.path.join('..', 'data', 'house_tiny.csv')#join函数会补充os的分隔符
data = pd.read_csv(data_file)
print(data)

inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)#dummy_na参数会处理带有NaN的值的哑变量
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print(X,y)


