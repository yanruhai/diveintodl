import sys

import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class LSTMScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input node

@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Initial state with shape: (batch_size, num_hiddens)
        H = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
        C = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
    else:
        H, C = H_C
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, self.W_xi) +
                        torch.matmul(H, self.W_hi) + self.b_i)#计算It=sigmoid(XtWxi+Ht-1Whi+bi)
        F = torch.sigmoid(torch.matmul(X, self.W_xf) +
                        torch.matmul(H, self.W_hf) + self.b_f)#计算Ft=sigmoid(XtWxf+Ht-1Whf+bf)
        O = torch.sigmoid(torch.matmul(X, self.W_xo) +
                        torch.matmul(H, self.W_ho) + self.b_o)#计算Ot=sigmoid(XtWxo+Ht-1Who+bo)
        C_tilde = torch.tanh(torch.matmul(X, self.W_xc) +
                           torch.matmul(H, self.W_hc) + self.b_c)#计算~C=tanh(XtWxc+Ht-1Whc+bc)
        C = F * C + I * C_tilde#Ct=Ft*Ct-1+It*C_tilde
        H = O * torch.tanh(C)#Ht=Ot*tanh(Ct)
        outputs.append(H)#将所有输出拼接成一个列表
    return outputs, (H, C)

class MinPerplexity:

    def start_new_stage(self,name):
        self.name=name
        self.min_train_loss = sys.float_info.max
        self.min_val_loss = sys.float_info.max
        self.min_train_name = ''
        self.min_val_name = ''


    def __init__(self):
        self.min_train_loss=sys.float_info.max
        self.min_val_loss=sys.float_info.max
        self.min_train_name = ''
        self.min_val_name=''
        self.all_min_train_loss = sys.float_info.max
        self.all_min_val_loss = sys.float_info.max
        self.name = ''

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


mp=MinPerplexity()

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


batch_size_list=[512,1024,2048]
num_hiddens_list=[32,64,128,256]
num_steps_list=[32,64,128]
lr_list=[2,3,4,5]
max_epochs_list=[50,100]
for batch_size in batch_size_list:
    for num_hiddens in num_hiddens_list:
        for num_steps in num_steps_list:
            for lr in lr_list:
                for max_epochs in max_epochs_list:
                    mp.start_new_stage(f"batch_size={batch_size},num_hiddens={num_hiddens},num_steps={num_steps},lr={lr},max_epochs={max_epochs}")
                    data = d2l.TimeMachine(batch_size=batch_size, num_steps=num_steps)#数据
                    lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=num_hiddens)#rnn对象
                    model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=lr)#model对象
                    model.plot = lambda *args, **kwargs: None  # 禁用绘图，None是函数体
                    trainer = d2l.Trainer(max_epochs=max_epochs, gradient_clip_val=1, num_gpus=1)#train对象
                    trainer.fit(model, data)#精简实现暂时注释
                    mp.print_train_val()

mp.print_all()
                    #plt.show()

#concise implementation
class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens,num_layers):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens,num_layers)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)


lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32,num_layers=2)
model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
#trainer.fit(model, data)
#plt.show()