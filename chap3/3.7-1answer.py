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



data = Data(num_train=100, num_val=30, num_inputs=20, batch_size=30)

class WeightDecay(d2l.LinearRegression):#这个类目的是使用库函数，减少代码长度，简化前面那个类。
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},#wd就是lamda
            {'params': self.net.bias}], lr=self.lr)

lam_list=np.arange(1,5,step=0.2)
train_list=[]
val_list=[]
for lam in lam_list:
    trainer = d2l.Trainer(max_epochs=10)
    model = WeightDecay(wd=3, lr=0.01)
    trainer.fit(model, data)
    model.eval()
    with torch.no_grad():
        # 训练集
        train_X = data.X[:data.num_train]
        train_y_true = data.y[:data.num_train]
        train_y_pred = model(train_X)
        train_diff = train_y_true - train_y_pred
        train_mse = float(torch.mean(train_diff ** 2))
        train_list.append(train_mse)
        # 验证集
        valid_X = data.X[data.num_train:]
        valid_y_true = data.y[data.num_train:]
        valid_y_pred = model(valid_X)
        valid_diff = valid_y_true - valid_y_pred
        valid_mse = float(torch.mean(valid_diff ** 2))
        val_list.append(valid_mse)
plt.figure(figsize=(10, 6))
plt.plot(lam_list,train_list)
plt.xlabel('lambd Index')
plt.ylabel('train diff')
plt.title('(Training Set)')
plt.legend()
plt.figure(figsize=(10, 6))
plt.plot(lam_list,val_list)
plt.xlabel('lambd Index')
plt.ylabel('validate diff')
plt.title('(validate Set)')
plt.legend()
plt.show()


