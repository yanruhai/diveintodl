import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

print('5.2.1. Implementation from Scratch')
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))

print('5.2.1.2. Model')
def relu(X):#ReLU激活函数
    a = torch.zeros_like(X)
    return torch.max(X, a)

@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)
    return torch.matmul(H, self.W2) + self.b2

def main():
    print('5.2.1.3. Training')
   # model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
    data = d2l.FashionMNIST(batch_size=256)
    trainer = d2l.Trainer(max_epochs=10)
    #trainer.fit(model, data)

    model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
    trainer.fit(model, data)
    plt.show()
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()



