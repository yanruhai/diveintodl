import time

import matplotlib
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()#将 matplotlib 的图形输出格式设置为 SVG
matplotlib.use('TkAgg')
print('4.2.1. Loading the Dataset')
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
data = FashionMNIST(resize=(32, 32))
print(len(data.train), len(data.val))
print(data.train.data.shape)#torch.Size([60000, 28, 28])
print(data.train[0][0].shape)
print(data.val.data.shape)#torch.Size([10000, 28, 28])
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',#pullover 套衫
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']#sandal:凉鞋 ankleboot:踝鞋
    return [labels[int(i)] for i in indices]

print('4.2.2. Reading a Minibatch')

@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
#指定用于数据加载的子进程数量（worker processes），以并行加载和预处理数据
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)

tic = time.time()
for X, y in data.train_dataloader():
    continue
print(f'{time.time() - tic:.2f} sec')

print('4.2.3. Visualization')

@d2l.add_to_class(FashionMNIST)  #@save
def show_images(self, imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if img.shape[0] == 1:
            img = img.squeeze(0)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show(block=True)  # 保持窗口打开
    return axes

@d2l.add_to_class(FashionMNIST)
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
if __name__ == '__main__':
    data = FashionMNIST(resize=(32, 32))
    batch = next(iter(data.val_dataloader()))
    data.visualize(batch)