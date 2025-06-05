import matplotlib.pyplot as plt
from d2l import torch as d2l
import torchvision
from torchvision import transforms
import torch


# 实现 show_images 函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
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
    plt.show()
    return axes


# 定义 FashionMNIST 类
class FashionMNIST(d2l.DataModule):
    def __init__(self, batch_size=64, resize=(28, 28), num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=False)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=False)

    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                           num_workers=self.num_workers)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def visualize(self, batch, nrows=6, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)


# 主程序
if __name__ == '__main__':
    data = FashionMNIST(resize=(32, 32))
    batch = next(iter(data.val_dataloader()))
    data.visualize(batch)