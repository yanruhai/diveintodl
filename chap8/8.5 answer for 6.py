import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l


def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())

class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, strides=4, padding=0),#输出尺寸 54*54
            nn.MaxPool2d(3, stride=2),#输出尺寸 26*26
            nin_block(256, kernel_size=5, strides=1, padding=2),#output_size=26*26
            nn.MaxPool2d(3, stride=2),#output_size=12*12
            nin_block(384, kernel_size=3, strides=1, padding=1),#output_size=12*12
            nn.LazyBatchNorm2d(),nn.Dropout(0.3), nn.MaxPool2d(3, stride=2),#output_size=5*5
            nin_block(num_classes, kernel_size=3, strides=1, padding=1),#output_size=5*5
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
        self.net.apply(d2l.init_cnn)

    def deep_count(self,seq_instance):
        for ss in seq_instance:
            if isinstance(ss,nn.Sequential):
                self.deep_count(ss)
            else:
                if isinstance(ss, (nn.Conv2d , nn.Linear)):
                    self.parameter_num += torch.numel(ss.weight)

    def forward(self, X):
        t = super().forward(X)
        self.parameter_num = 0
        self.deep_count(self.net)
        print("parameters:", self.parameter_num)  # 统计参数的数量
        return t

def main():
    # 清空 GPU 缓存
    torch.cuda.empty_cache()
    print(f"初始内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    model = NiN(lr=0.05)
    print(f"模型加载后内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    trainer = d2l.Trainer(max_epochs=30, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)

    print(f"峰值内存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"缓存内存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    #next,iter是内置函数
    trainer.fit(model, data)
    print(f"峰值内存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"缓存内存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    plt.show()



if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()