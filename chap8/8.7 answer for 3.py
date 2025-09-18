import time

import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):#第一个参数是卷积核的数量，第二个是输出通道，但是这个值会累加，也称为成长率
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X

blk = DenseBlock(2, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)

def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


@d2l.add_to_class(DenseNet)
def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4),
             lr=0.1, num_classes=10):
    super(DenseNet, self).__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(self.b1())
    for i, num_convs in enumerate(arch):
        self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
                                                          growth_rate))
        # The number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(arch) - 1:#如果不是最后一次
            num_channels //= 2#整除
            self.net.add_module(f'tran_blk{i+1}', transition_block(
                num_channels))
    self.net.add_module('last', nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.LazyLinear(num_classes)))
    self.net.apply(d2l.init_cnn)
def main():
    # 清空 GPU 缓存
        torch.cuda.empty_cache()
        print(f"初始内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

        start_time=time.time()
        model = DenseNet(lr=0.01)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 关键: 移动模型到 GPU
        print(f"模型加载后内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
        data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
        trainer.fit(model, data)
        print(f"前向传播后内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"峰值内存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"缓存内存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        elapse_time=time.time()-start_time
        print('elapse:',elapse_time)
        plt.show()

if __name__ == "__main__":  # 确保代码块只在直接运行时执行，而在被导入时不执行
    main()