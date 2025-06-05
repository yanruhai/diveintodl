import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)#三元表达式
        #slice(self.num_train, None)表示从num_train到结尾
        return self.get_tensorloader([self.X, self.y], train, i)

print('3.7.3. Implementation from Scratch')
def l2_penalty(w):
    return (w ** 2).sum() / 2

def l1_penalty(w):
    return torch.abs(w).sum()

class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.lambd * l1_penalty(self.w)

data = Data(num_train=1000, num_val=1000, num_inputs=20, batch_size=30)
trainer = d2l.Trainer(max_epochs=50)

lambd =0.1
model = WeightDecayScratch(num_inputs=20, lambd=lambd, lr=0.01)
model.board.yscale='log'#y 轴的刻度是对数形式的（例如  10^{-2}, 10^{-1}, 10^0, 10^1
trainer.fit(model, data)


plt.show()