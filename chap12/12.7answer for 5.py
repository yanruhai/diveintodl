import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

def init_cnn(module):  #@save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:#偏置初始化为0
        nn.init.xavier_uniform_(module.weight)

class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.ReLU(),
            nn.LazyLinear(84), nn.ReLU(),
            nn.LazyLinear(num_classes)
        )

@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = torch.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

def main():
    criterion = nn.CrossEntropyLoss()
    data = d2l.FashionMNIST(batch_size=128)
    model = LeNet(lr=0.1)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
    #true代表是否是训练,get_dataloader得到：返回的迭代器每次迭代会产生一个 元组 (images, labels)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.003)
    num_epochs=10
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        for batch in data.get_dataloader(True):
            outputs=model(*batch[:-1])
            l = criterion(outputs, batch[-1])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            total_loss += l.item()
            num_batches += 1
            # 计算准确率
            predictions = torch.argmax(outputs, dim=1)  # 获取预测的类别
            correct = (predictions == batch[-1]).sum().item()
            total = batch[-1].size(0)
            accuracy = correct / total

            print(f'Loss: {l.item():.4f}, Accuracy: {accuracy:.4f}')
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')


if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()