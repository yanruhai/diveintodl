import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  # @save
    """The Residual block of ResNet models."""

    def __init__(self, input_channels, out_channel1, out_channel2, out_channel3,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 主路径卷积层
        self.conv1 = nn.Conv2d(input_channels, out_channel1, kernel_size=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channel1, out_channel2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channel2, out_channel3, kernel_size=1)

        # 批量归一化层（每个卷积后都需要）
        self.bn1 = nn.BatchNorm2d(out_channel1)
        self.bn2 = nn.BatchNorm2d(out_channel2)
        self.bn3 = nn.BatchNorm2d(out_channel3)

        # 残差连接的1x1卷积（用于通道和尺寸匹配）
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, out_channel3, kernel_size=1, stride=strides)
        else:
            self.conv4 = None

    def forward(self, X):
        # 主路径前向传播（卷积->归一化->激活）
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))  # 最后一个卷积后不立即激活

        # 残差连接处理
        if self.conv4:
            X = self.conv4(X)

        # 残差相加后再激活
        Y += X
        return F.relu(Y)


class ResidualNet152(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super(ResidualNet152, self).__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i + 2}', self.block(*b, first_block=(i == 0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)

    def b1(self):
        """第一个卷积块：7x7卷积 + 最大池化 + 初始残差块"""
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # FashionMNIST是单通道
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 初始残差块：64->64->64->256
            Residual(64, 64, 64, 256, use_1x1conv=True, strides=1)
        )

    def block(self, num_residuals, input_channel, out_channel1, out_channel2, out_channel3, first_block=False):
        """构建由多个残差块组成的大模块"""
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                # 非第一个大模块的第一个残差块：需要下采样和通道调整
                blk.append(Residual(
                    input_channel, out_channel1, out_channel2, out_channel3,
                    use_1x1conv=True, strides=2
                ))
            else:
                # 其他残差块：输入通道与上一个残差块输出通道一致
                blk.append(Residual(
                    out_channel3, out_channel1, out_channel2, out_channel3,
                    use_1x1conv=False, strides=1
                ))
        return nn.Sequential(*blk)


class ResNet(ResidualNet152):
    def __init__(self, lr=0.1, num_classes=10):
        # 架构参数：(残差块数量, 输入通道, 中间通道1, 中间通道2, 输出通道)
        super().__init__((
            (2, 256, 64, 64, 256),  # conv2_x (实际包含2+1=3个残差块，因为b1中已有1个)
            (8, 256, 128, 128, 512),  # conv3_x
            (36, 512, 256, 256, 1024),  # conv4_x
            (3, 1024, 512, 512, 2048)  # conv5_x
        ), lr, num_classes)


def main():
    # 启用异常检测帮助调试梯度问题
    torch.autograd.set_detect_anomaly(True)

    model = ResNet(lr=0.01)
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model, data)
    plt.show()


if __name__ == "__main__":
    main()
