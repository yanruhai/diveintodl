import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
import os

os.environ["OMP_NUM_THREADS"] = "16"  # 设置线程数匹配 CPU
os.environ["MKL_NUM_THREADS"] = "16"  # Intel MKL 线程优化
torch.set_num_threads(16)  # 假设CPU有4核
print('5.2.1. Implementation from Scratch')
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs,num_h2, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_h2) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_h2))
        self.W3 = nn.Parameter(torch.randn(num_h2,num_outputs)*sigma)
        self.b3=nn.Parameter(torch.zeros(num_outputs))

print('5.2.1.2. Model')
def relu(X):#ReLU激活函数
    a = torch.zeros_like(X)
    return torch.max(X, a)

@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)
    H2=relu(torch.matmul(H,self.W2)+self.b2)
    return torch.matmul(H2, self.W3) + self.b3

def main2():
    start_time = time.time()  # 记录开始时间
   # model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
    data = d2l.FashionMNIST(resize=(32,32),batch_size=256)
    #trainer.fit(model, data)
    train_loss_list=[]
    val_loss_list=[]
    trainer = d2l.Trainer(max_epochs=38)
    model = MLPScratch(num_inputs=1024, num_outputs=10, num_h2=128,num_hiddens=128,lr=0.1)
    trainer.fit(model, data)
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time
    print('执行时间',execution_time)
    plt.show()

class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main2()



