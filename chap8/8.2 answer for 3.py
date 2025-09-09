import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l



class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        conv_blks = []
        for (num_convs, out_channels) in arch:
            if num_convs==0:
                conv_blks.append(self.vgg_block_special(out_channels))
            else:
                conv_blks.append(self.vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)

    def vgg_block(self,num_convs, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)

    def vgg_block_special(self,out_channels):
        layers=[]
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.LazyConv2d(out_channels, kernel_size=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

def main():
   # VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
       # (1, 1, 224, 224))

    modelVGG16 = VGG(arch=((2, 64), (2, 128), (0, 256), (0, 512), (0, 512)), lr=0.01)
    trainer = d2l.Trainer(max_epochs=10,num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
    modelVGG16.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(modelVGG16, data)
    plt.show()

if __name__ == '__main__':
    main()

