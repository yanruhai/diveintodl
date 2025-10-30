import torch
from d2l.torch import d2l
from matplotlib import pyplot as plt
from torch import nn

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

class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens,num_layers):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens,num_layers)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)


class RNNRegression(d2l.Module):
    def __init__(self, rnn, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(
                self.rnn.num_hiddens, 1) * 0.01)  # 用于输出的Whq
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

    def forward(self, X, H_C=None):
        #embs = X.unsqueeze(2)
        embs=X.permute(1,0,2)
        rnn_outputs, _ = self.rnn(embs, H_C)  # rnn_outputs是[h1,h2,...,ht]列表,hi∈(batch_size, num_hiddens)
        return self.output_layer(rnn_outputs)

num_steps=4
data = Data()
lstm = LSTM(num_inputs=num_steps, num_hiddens=32,num_layers=2)
model = RNNRegression(lstm, lr=4)
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)#train对象
trainer.fit(model, data)
plt.show()