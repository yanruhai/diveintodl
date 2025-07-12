import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, dropout):#dropout是概率值，需要去掉的神经元概率
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()#不可转成int
    return mask * X / (1.0 - dropout)#h/1-p

X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))

print('5.6.2.1. Defining the Model')
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)#只需输出维度
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()
        self.H1_val=0#记录h1的方差
        self.H2_val=0

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        #if self.training:
        #    H1 = dropout_layer(H1, self.dropout_1)
        self.H1_val += torch.var(H1.detach())
        H2 = self.relu(self.lin2(H1))

        #if self.training:
        #    H2 = dropout_layer(H2, self.dropout_2)
        self.H2_val += torch.var(H2.detach())
        return self.lin3(H2)

    def clear_h_var(self):
        self.H1_val=0
        self.H2_val=0

class My_Trainer(d2l.Trainer):
    def __init__(self,max_epochs=30):
        super().__init__(max_epochs)
        self.H1_val_list=[]
        self.H2_val_list=[]

    def fit_epoch(self):
        super().fit_epoch()
        self.after_epoch()

    def after_epoch(self):
        model = self.model
        self.H1_val_list.append(model.H1_val)
        self.H2_val_list.append(model.H2_val)
        model.clear_h_var()






def main():
    hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
               'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
    model = DropoutMLPScratch(**hparams)#**是解包操作，将字典转成参数
    data = d2l.FashionMNIST(batch_size=256)
    trainer = My_Trainer(max_epochs=30)
    epochs=range(30)
    trainer.fit(model, data)
    plt.figure()
    plt.plot(epochs,trainer.H1_val_list,label='h1_var')
    plt.plot(epochs,trainer.H2_val_list,label='h2_var')
    plt.legend()
    plt.show()

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()

print('5.6.3. Concise Implementation')






