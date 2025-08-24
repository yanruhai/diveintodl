#7.6 answer for 1,2
import time

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
        self.total_time=0
        self.total_image=0#图片的总张数
        self.net = nn.Sequential(
            nn.LazyConv2d(10, kernel_size=7, padding=2), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(22, kernel_size=5), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(30, kernel_size=3), nn.Sigmoid(),
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


@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    """Defined in :numref:`sec_linear_scratch`"""
    start_time=time.time()
    self.model.train()
    cur_batch=0
    for batch in self.train_dataloader:
        processed_batch = self.prepare_batch(batch)
        batch_size = processed_batch[0].shape[0]
        cur_batch+=batch_size
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        processed_batch = self.prepare_batch(batch)
        batch_size = processed_batch[0].shape[0]  # 验证集批次大小
        cur_batch+=batch_size
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
    end_time=time.time()
    elape_time=end_time-start_time
    print('本次运行时长',elape_time)
    print('本次批次量',cur_batch)
    self.model.total_time+=elape_time
    self.model.total_image += cur_batch





def main():
    model = LeNet()
    model.layer_summary((1, 1, 28, 28))

    trainer = d2l.Trainer(max_epochs=10)
    data = d2l.FashionMNIST(batch_size=128)
    model = LeNet(lr=0.1)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)#初始化，前向一次
    #true代表是否是训练,get_dataloader得到：返回的迭代器每次迭代会产生一个 元组 (images, labels)
    trainer.fit(model, data)
    print('单位时间处理图片数量:',model.total_image/model.total_time)
    plt.show()

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()