import time
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l

# ResNeXtBlock 定义，修复 groups 参数
class ResNeXtBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False, strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        # 确保 bot_channels 足够大且可被 groups 整除
        bot_channels = max(bot_channels, groups)  # 防止 bot_channels 过小
        if bot_channels % groups != 0:
            bot_channels = (bot_channels // groups + 1) * groups  # 调整为 groups 的倍数
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3, stride=strides, padding=1, groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)

# 过渡层
def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

# 修改后的 DenseNet，使用 ResNeXtBlock
class DenseNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def __init__(self, num_channels=64, groups=8, bot_mul=0.5, arch=(4, 4, 4, 4), lr=0.1, num_classes=10):
        super(DenseNet, self).__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1())
        for i, num_blocks in enumerate(arch):
            for j in range(num_blocks):
                self.net.add_module(
                    f'resnext_blk{i+1}_{j+1}',
                    ResNeXtBlock(num_channels=num_channels, groups=groups, bot_mul=bot_mul, use_1x1conv=(j==0 and i!=0), strides=1 if j!=0 else 2)
                )
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', transition_block(num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)

# 主函数
def main():
    start_time = time.time()
    model = DenseNet(lr=0.01, groups=8, bot_mul=0.5, arch=(4, 4, 4, 4))
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
    trainer.fit(model, data)
    elapse_time = time.time() - start_time
    print('Elapsed time:', elapse_time)
    plt.show()

if __name__ == "__main__":
    main()