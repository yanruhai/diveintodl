import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = torch.arange(1, T + 1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2

data = Data()
d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

@d2l.add_to_class(Data)
def get_dataloader(self, train):
    features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]#生成T*4的张量
    self.features = torch.stack(features, 1)#该张量在列上堆
    self.labels = self.x[self.tau:].reshape((-1, 1))#目标列生成
    i = slice(0, self.num_train) if train else slice(self.num_train, None)#获得训练或验证模式下标
    return self.get_tensorloader([self.features, self.labels], train, i)