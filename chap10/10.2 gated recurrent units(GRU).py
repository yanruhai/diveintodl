import sys
import time

import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l


class MinPerplexity:

    def start_new_stage(self,name):
        self.name=name
        self.min_train_loss = sys.float_info.max
        self.min_val_loss = sys.float_info.max
        self.min_train_name = ''
        self.min_val_name = ''




    def __init__(self):
        self.all_min_train_loss = sys.float_info.max
        self.all_min_val_loss = sys.float_info.max
        self.start_new_stage('第一次运行')

    def save_min(self,cur_loss,train):
        if train:
            if cur_loss<self.min_train_loss:
                self.min_train_loss=cur_loss
                if cur_loss<self.all_min_train_loss:
                    self.min_train_loss=cur_loss
                    self.min_train_name=self.name
        else:
            if cur_loss<self.min_val_loss:
                self.min_val_loss=cur_loss
                if cur_loss < self.all_min_val_loss:
                    self.min_val_loss = cur_loss
                    self.min_val_name = self.name


    def print_train_val(self):
        print(self.name,'最小训练损失:',self.min_train_loss,'最小验证损失:',self.min_val_loss)

    def print_all(self):
        print(f"在{self.min_train_name}下全局最小训练损失:{self.all_min_train_loss},在{self.min_val_name}下全局最小验证损失：{self.all_min_val_loss}")

class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
    #让初始权重值集中在 0 附近的一个小范围（大部分值会落在 [-0.03, 0.03] 之间，因为正态分布中约 99.7% 的数据落在均值 ±3 倍标准差范围内）
        #sigma是标准差
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state

@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    if H is None:
        # Initial state with shape: (batch_size, num_hiddens)
        H = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, self.W_xz) +
                        torch.matmul(H, self.W_hz) + self.b_z)#更新门
        R = torch.sigmoid(torch.matmul(X, self.W_xr) +
                        torch.matmul(H, self.W_hr) + self.b_r)#复位门
        H_tilde = torch.tanh(torch.matmul(X, self.W_xh) +
                           torch.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H)
    return outputs, H

@d2l.add_to_class(d2l.RNNLMScratch)
def training_step(self, batch):
    l = self.loss(self(*batch[:-1]), batch[-1])
    self.plot('ppl', d2l.exp(l), train=True)
    mp.save_min(l.detach().item(),True)
    return l

@d2l.add_to_class(d2l.RNNLMScratch)
def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
        mp.save_min(l.detach().item(),False)

data = d2l.TimeMachine(batch_size=1024, num_steps=32)
gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
mp=MinPerplexity()
model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)#测试精简模式，暂时不用
#plt.show()

class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens)




batch_size_list=[1024]
num_hiddens_list=[128]
num_steps_list=[32]
lr_list=[1]
max_epochs_list=[50]

for batch_size in batch_size_list:
    for num_hiddens in num_hiddens_list:
        for num_steps in num_steps_list:
            for lr in lr_list:
                for max_epochs in max_epochs_list:
                    mp.start_new_stage(f"batch_size={batch_size},num_hiddens={num_hiddens},num_steps={num_steps},lr={lr},max_epochs={max_epochs}")
                    start_time = time.time()
                    data = d2l.TimeMachine(batch_size=batch_size, num_steps=num_steps)
                    gru = GRU(num_inputs=len(data.vocab), num_hiddens=num_hiddens)
                    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=lr)
                    #model.plot = lambda *args, **kwargs: None  # 禁用绘图，None是函数体
                    trainer = d2l.Trainer(max_epochs=max_epochs, gradient_clip_val=1, num_gpus=1)
                    trainer.fit(model, data)
                    mp.print_train_val()
                    end_time=time.time()
                    print("运行时间为：",end_time-start_time)
                    plt.show()
mp.print_all()
