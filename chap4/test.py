import matplotlib.pyplot as plt
from d2l import torch as d2l
import torchvision
from torchvision import transforms

# 定义 FashionMNIST 类
class FashionMNIST(d2l.DataModule):
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=False)

# 加载原始尺寸 (28x28) 和调整尺寸 (32x32) 的数据集
data_28 = FashionMNIST(resize=(28, 28))
data_32 = FashionMNIST(resize=(32, 32))

# 获取同一张图像
image_28, label = data_28.train[0]
image_32, _ = data_32.train[0]

# 类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 显示两张图像
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_28.squeeze(), cmap='gray')
plt.title(f'28x28, Label: {class_names[label]}')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_32.squeeze(), cmap='gray')
plt.title(f'32x32, Label: {class_names[label]}')
plt.axis('off')
plt.tight_layout()
plt.show()