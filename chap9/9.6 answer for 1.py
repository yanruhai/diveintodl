import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4,num_steps=32):
        self.save_hyperparameters()
        self.time = torch.arange(1, T + 1, dtype=torch.float32)#1->1001
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2
#每行4列数据做预测
    def get_dataloader(self, train):
        #features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]#生成T*4的张量
        t1 = [self.x[i:i + self.num_steps + self.tau] for i in range(self.T - self.num_steps - self.tau+1)]
        t1 = torch.stack(t1, 0)
        t2 = [t1[:, i:i + self.tau] for i in range(len(t1[0]) - self.tau)]
        t2 = torch.stack(t2, 1)
        label = [t1[:, self.tau:]]
        self.labels = torch.stack(label, 2)#目标列生成
        self.features = t2
        i = slice(0, self.num_train) if train else slice(self.num_train, None)#获得训练或验证模式下标,用None是表示到结尾
        return self.get_tensorloader([self.features, self.labels], train, i)


#plt.show()
batch_size=16
tau=8
data = Data(batch_size=batch_size,tau=tau)
for batch in data.get_dataloader(True):
    print(batch)

class MyLinearRegression(d2l.LinearRegression):
    def __init__(self,lr,num_hiddens,batch_size,sigma=0.01):
        super().__init__(lr)
        self.save_hyperparameters()
        ll=[nn.LazyLinear(num_hiddens) for _ in range(batch_size)]
        self.layers=nn.ModuleList(ll)
        self.W_xh = nn.Parameter(
            torch.randn(tau, num_hiddens) * sigma)#输入维度固定为4
        self.W_hh = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

    def forward(self, inputs,state=None):
        if state is None:
            state=torch.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs),num_inputs usually is vacab_size
            # 按照第一维循环，每次读取 (batch_size,num_inputs).依次计算h1,h2...
            state = torch.tanh(torch.matmul(X, self.W_xh) +
                               torch.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)  # 产生[h1,h2,...,ht]列表，hi∈(batch_size, num_hiddens)
        return outputs, state  # 继续新的批次可能需要当前的ht作为新批次的h0，这里会包装成元组

class RNNRegression(d2l.Module):
    def __init__(self, rnn, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(
                self.rnn.num_hiddens, 1) * self.rnn.sigma)  # 用于输出的Whq
        self.b_q = nn.Parameter(torch.zeros(1))

    def loss(self, y_hat, y):
        l=nn.functional.mse_loss(y_hat,y)
        return l

    def training_step(self, batch):
        # batch是list[2]，如果传入batch[0]会导致为[x]数据，格式错误，所以需要解包
        l = self.loss(self(*batch[:-1]), batch[-1])  # 前向后得到outputs,state,默认处理第一项
        self.plot('ppl', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', l, train=False)

    def output_layer(self, rnn_outputs):
        '''将数据经过与矩阵Whq相乘得到输出'''
        # 得到维度为batch_size*one_hot张量的列表，列表按时间步排列.rnn_outputs是[h1,h2,...,ht]列表,hi∈(batch_size, num_hiddens)
        outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return torch.stack(outputs, 1)  # 将时间步放在第一维(序数是2)，堆起来，得到(batch_size*time_steps*one_hot)张量

    def forward(self, X, state=None):
        #embs = X.unsqueeze(2)
        embs=X.permute(1,0,2)
        rnn_outputs, _ = self.rnn(embs, state)  # rnn_outputs是[h1,h2,...,ht]列表,hi∈(batch_size, num_hiddens)
        return self.output_layer(rnn_outputs)

    def k_step_prediction(self, pre_data, len_predict):  # len_predict是需要预测的长度
        out_puts = []  # 预测结果
        #out_puts.append(pre_data[0][0].detach().numpy())
        temp = torch.unsqueeze(torch.unsqueeze(pre_data[0],0),0)
        state=None
        for idx in range(tau,len_predict):
            rnn_output,state = self.rnn(temp,state)
            if idx < len(pre_data)-1:  # warm-up stage
                out_puts.append((pre_data[idx][pre_data.shape[1]-1]).item())
                temp = torch.unsqueeze(pre_data[idx], 0)
            else:
                out=self.output_layer(rnn_output)
                out_puts.append(out.detach().item())
                temp = torch.unsqueeze(torch.unsqueeze(torch.tensor(out_puts[-tau:]),0),0)#最后的4位数据继续做下次预测
        return out_puts

#d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
rnn = MyLinearRegression(lr=0.01,num_hiddens=128,batch_size=batch_size)
model=  RNNRegression(rnn)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)




start_data=[torch.tensor(data.x[i:i+tau]) for i in range(0,600)]
start_data=torch.stack(start_data)
y_pre=model.k_step_prediction(start_data,1000)
onestep_preds = model(data.features).detach()
y1=data.labels.squeeze()
y2=onestep_preds.squeeze()
y1_1=y1[:,0]
y1_2=y1[len(y1)-1,1:]
y1=torch.cat((y1_1,y1_2))#去掉重复的数据
y2_1=y2[:,0]
y2_2=y2[len(y2)-1,1:]
y2=torch.cat((y2_1,y2_2))
y1=y1.numpy()
y2=y2.numpy()
y_pre=np.array(y_pre)
time_list = data.time[data.tau:].detach().numpy()
d2l.plot(time_list, [y1,y_pre], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
plt.show()
multistep_preds = torch.zeros(data.T)
multistep_preds[:] = data.x
for i in range(data.num_train + data.tau, data.T):#num_train=600,tau=4,T=1000
    multistep_preds[i] = model(multistep_preds[i - data.tau:i].reshape((1, -1)))
multistep_preds = multistep_preds.detach().numpy()

d2l.plot([data.time[data.tau:], data.time[data.num_train+data.tau:]],
         [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time',
         'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))

def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model(torch.stack(features[i : i+data.tau], 1))
        features.append(preds.reshape(-1))
        #如果原始张量是三维形状 (a, b, c)，使用 reshape(-1) 后，结果会是1 维张量，形状为 (a×b×c,)
    return features[data.tau:]

steps = (2,4, 16, 64)
preds = k_step_pred(steps[-1])#预测出的结果,64个列表，每个列表(933,)
mm=[preds[k - 1].detach().numpy() for k in steps]
d2l.plot(data.time[data.tau+steps[-1]-1:],
         [preds[k - 1].detach().numpy() for k in steps], 'time', 'x',
         legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
plt.show()