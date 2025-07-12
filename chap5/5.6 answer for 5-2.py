import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,num_hiddens_3,num_hiddens_4,
                 dropout_1, dropout_2,dropout_3,dropout_4, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_hiddens_3), nn.ReLU(),
            nn.Dropout(dropout_3), nn.LazyLinear(num_hiddens_4), nn.ReLU(),
            nn.Dropout(dropout_4), nn.LazyLinear(num_outputs))


def main():
    hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,'num_hiddens_3':128,
               'num_hiddens_4':256,
               'dropout_1':0.3, 'dropout_2':0.2,'dropout_3':0.3,'dropout_4':0.2, 'lr':0.1}
    model = DropoutMLP(**hparams)#**是解包操作，将字典转成参数
    data = d2l.FashionMNIST(batch_size=256)
    trainer = d2l.Trainer(max_epochs=50)
    trainer.fit(model, data)
    plt.show()

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()
