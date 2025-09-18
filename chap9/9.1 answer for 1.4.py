import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l



class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = torch.arange(1, T + 1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2

#每行4列数据做预测
    def get_dataloader(self, train):
        features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]#生成T*4的张量
        self.features = torch.stack(features, 1)#该张量在列上堆,996*4的张量
        features_squ = self.features ** 2
        features_tri = self.features ** 3
        self.features = torch.cat([self.features, features_squ,features_tri], dim=1)
        self.labels = self.x[self.tau:].reshape((-1, 1))#目标列生成
        i = slice(0, self.num_train) if train else slice(self.num_train, None)#获得训练或验证模式下标
        return self.get_tensorloader([self.features, self.labels], train, i)


data = Data(tau=4)
d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
#plt.show()


model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)


onestep_preds = model(data.features).detach().numpy()
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))



multistep_preds = torch.zeros(data.features.shape)
multistep_preds[:] = data.features

print("linear model para:",model.get_w_b())

for i in range(data.num_train + data.tau, data.T-data.tau):#num_train=600,tau=4,T=1000
    #print(multistep_preds[i-data.tau])
    t = model(multistep_preds[i-data.tau])
    t_squ=t**2
    t_tri=t**3
    orgin_tensor=torch.tensor([t,t_squ,t_tri])
    new_tensor=orgin_tensor.unsqueeze(0)#转成1*3的张量
    for ind in range(data.tau):
        multistep_preds[i-ind,ind*3:ind*3+3] = new_tensor


multistep_preds = multistep_preds[:,0].detach().numpy()

d2l.plot([data.time[data.tau:], data.time[data.num_train+data.tau:data.T-data.tau]],
         [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time',
         'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))

def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model(torch.stack(features[i : i+data.tau*3], 1))
        features.append(preds.reshape(-1))
        #如果原始张量是三维形状 (a, b, c)，使用 reshape(-1) 后，结果会是1 维张量，形状为 (a×b×c,)
    return features[data.tau*3:]


tau_list=range(20)


'''steps = (2,4, 16, 64)
preds = k_step_pred(steps[-1])#预测出的结果,64个列表，每个列表(933,)
d2l.plot(data.time[data.tau+steps[-1]-1:],
         [preds[k - 1].detach().numpy() for k in steps], 'time', 'x',
         legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))'''
plt.show()