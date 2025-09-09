import time
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

N=7#用于论文内的n的初值
PADDING=int((N-1)/2)

class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):#这里的b1-b4的格式是(batch,channels,height,width),最后一行在dim=1上的cat就是在channels上连接
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)

class Modified_Inception_C1(nn.Module):
   def __init__(self, b1_m3, b2_m2, b3, b4, **kwargs):
        super().__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(b1_m3[0], kernel_size=1)
        self.b1_2 = nn.LazyConv2d(b1_m3[1], kernel_size=3, padding=1)
        self.b1_3 = nn.LazyConv2d(b1_m3[2], kernel_size=3, padding=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(b2_m2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(b2_m2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.b3_2 = nn.LazyConv2d(b3, kernel_size=5, padding=2,stride=1)
        # Branch 4
        self.b4_1 = nn.LazyConv2d(b4, kernel_size=1)

   def forward(self, x):
       b1 = F.relu(self.b1_1(x))
       b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
       b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
       b4 = F.relu(self.b4_1(x))
       return torch.cat((b1, b2, b3, b4), dim=1)


class Modified_Inception_C2(nn.Module):
   def __init__(self, b1_m5, b2_m3, b3, b4, **kwargs):
        super().__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(b1_m5[0], kernel_size=1)
        self.b1_2 = nn.LazyConv2d(b1_m5[1], kernel_size=(1, N), padding=(0, PADDING))
        self.b1_3 = nn.LazyConv2d(b1_m5[2], kernel_size=(N, 1), padding=(PADDING, 0))
        self.b1_4 = nn.LazyConv2d(b1_m5[3], kernel_size=(1, N), padding=(0, PADDING))
        self.b1_5 = nn.LazyConv2d(b1_m5[4], kernel_size=(N, 1), padding=(PADDING,0))

        # Branch 2
        self.b2_1 = nn.LazyConv2d(b2_m3[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(b2_m3[1], kernel_size=(1,N), padding=(0,PADDING))
        self.b2_3 = nn.LazyConv2d(b2_m3[2], kernel_size=(N,1), padding=(PADDING,0))
        # Branch 3
        self.b3_1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.b3_2 = nn.LazyConv2d(b3, kernel_size=1)
        # Branch 4
        self.b4_1 = nn.LazyConv2d(b4, kernel_size=1)

   def forward(self, x):
       b1 = F.relu(self.b1_5(F.relu(self.b1_4(F.relu(self.b1_3(F.relu(self.b1_2(F.relu(self.b1_1(x))))))))))
       b2 = F.relu(self.b2_3(F.relu(self.b2_2(F.relu(self.b2_1(x))))))
       b3 = F.relu(self.b3_2(self.b3_1(x)))
       b4 = F.relu(self.b4_1(x))
       return torch.cat((b1, b2, b3, b4), dim=1)


class Modified_Inception_C3(nn.Module):
   def __init__(self, b1_m2,b1_1,b1_2, b2,b2_1,b2_2, b3, b4, **kwargs):
        super().__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(b1_m2[0], kernel_size=1)
        self.b1_2 = nn.LazyConv2d(b1_m2[1], kernel_size=3, padding=1)
        self.b1_3_1=nn.LazyConv2d(b1_1, kernel_size=(1,3), padding=(0,1))
        self.b1_3_2=nn.LazyConv2d(b1_2,kernel_size=(3,1),padding=(1,0))
        # Branch 2
        self.b2_1 = nn.LazyConv2d(b2, kernel_size=1)
        self.b2_2_1 = nn.LazyConv2d(b2_1, kernel_size=(1,3), padding=(0,1))
        self.b2_2_2 = nn.LazyConv2d(b2_2, kernel_size=(3,1), padding=(1,0))
        # Branch 3
        self.b3_1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.b3_2 = nn.LazyConv2d(b3, kernel_size=1)
        # Branch 4
        self.b4_1 = nn.LazyConv2d(b4, kernel_size=1)

   def forward(self, x):
       temp_1=F.relu(self.b1_2(F.relu(self.b1_1(x))))
       b1_1 = F.relu(self.b1_3_1(temp_1))
       b1_2 = F.relu(self.b1_3_2(temp_1))
       temp_2=F.relu(self.b2_1(x))
       b2_1 = F.relu(self.b2_2_1(temp_2))
       b2_2 = F.relu(self.b2_2_2(temp_2))
       b3 = F.relu(self.b3_2(self.b3_1(x)))
       b4 = F.relu(self.b4_1(x))
       return torch.cat((b1_1,b1_2, b2_1,b2_2, b3, b4), dim=1)


class GoogleNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(32, kernel_size=3, stride=2),
            nn.ReLU(), nn.LazyConv2d(32, kernel_size=3),
            nn.ReLU(), nn.LazyConv2d(32, kernel_size=3),
            nn.ReLU(), nn.LazyConv2d(64, kernel_size=3, padding=1),
            nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2),
            nn.LazyConv2d(80,kernel_size=3,stride=2),
            nn.ReLU(),nn.LazyConv2d(192,kernel_size=3,stride=2),
            nn.ReLU(),nn.LazyConv2d(288,kernel_size=3),
            nn.ReLU())

@d2l.add_to_class(GoogleNet)
def b2(self):
    return nn.Sequential(Modified_Inception_C1(b1_m3=(64,64,64),b2_m2=(96,96),b3=32,b4=32),
                         Modified_Inception_C2(b1_m5=(64,64,64,64,64),b2_m3=(96,96,96),b3=32,b4=32),
                         Modified_Inception_C3(b1_m2=(64,64),b1_1=32,b1_2=32,b2=96,b2_1=48,b2_2=48,b3=32,b4=32)
       )

@d2l.add_to_class(GoogleNet)
def b3(self):
    return nn.Sequential(nn.MaxPool2d(kernel_size=8)
                         )



@d2l.add_to_class(GoogleNet)
def __init__(self, lr=0.1, num_classes=10):
    super(GoogleNet, self).__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), nn.AdaptiveAvgPool2d((1, 1)),  # 输出 (128, C_total, 1, 1)
        # 第三步：展平（去掉大小为1的维度）
        nn.Flatten(),  # 输出 (128, C_total)，符合全连接层输入要求
                             nn.LazyLinear(num_classes))
    self.net.apply(d2l.init_cnn)


def main():
    start_time=time.time()
    model = GoogleNet(lr=0.01)
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128, resize=(299, 299))
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model, data)
    end_time=time.time()
    print(end_time-start_time)#output is 163
    plt.show()

if __name__ == '__main__':
    main()