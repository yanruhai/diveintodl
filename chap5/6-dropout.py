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

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))

def main():
    hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
               'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
    model = DropoutMLPScratch(**hparams)
    data = d2l.FashionMNIST(batch_size=256)
    trainer = d2l.Trainer(max_epochs=10)
    #trainer.fit(model, data)

    model = DropoutMLP(**hparams)
    trainer.fit(model, data)
    plt.show()

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()

print('5.6.3. Concise Implementation')






