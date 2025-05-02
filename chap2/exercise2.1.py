import os

import pandas as pd

data_file = os.path.join('..', 'data', 'abalone.data')#join函数会补充os的分隔符
data = pd.read_csv(data_file)
#print(data.info())
print(data['M'].info())