import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.parameter_num=0
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)

    '''def deep_count(self, seq_instance):#计算参数量
        for ss in seq_instance:
            if isinstance(ss, nn.Sequential):
                self.deep_count(ss)
            else:
                if isinstance(ss, (nn.Conv2d, nn.Linear)):
                    self.parameter_num += torch.numel(ss.weight)

    def forward(self, X):
        t = super().forward(X)
        self.parameter_num = 0
        self.deep_count(self.net)
        print("parameters:", self.parameter_num)  # 统计参数的数量
        return t'''







def main():


    AlexNet().layer_summary((1, 1, 224, 224))

    model = AlexNet(lr=0.01)
    data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
    trainer = d2l.Trainer(max_epochs=10,num_gpus=2)
    trainer.fit(model, data)

    plt.show()

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()