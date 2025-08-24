import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(48, kernel_size=5, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(96, kernel_size=3, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)

def main():

    model = AlexNet(lr=0.01)
    data = d2l.FashionMNIST(batch_size=128)#28*28
    trainer = d2l.Trainer(max_epochs=10,num_gpus=2)
    trainer.fit(model, data)
    plt.show()

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()