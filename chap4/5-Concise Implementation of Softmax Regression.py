import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

print('4.5.1. Defining the Model')
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
print('4.5.2. Softmax Revisited')
@d2l.add_to_class(d2l.Classifier)  # @save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    Y = Y.reshape((-1,))
    return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

def main():
    data = d2l.FashionMNIST(batch_size=256)
    model = SoftmaxRegression(num_outputs=10, lr=0.1)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
    plt.show()

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()