import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

N=8096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("使用设备:", device)

A = torch.zeros(N, N)#.to(device)
B = torch.randn(N, N)#.to(device)
C = torch.randn(N, N)#.to(device)

class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    #假设你有一个数组 [a, b, c, d]，执行 cumsum 后的结果是：[a, a+b, a+b+c, a+b+c+d]
    def printAll(self):
        for t in self.times:
            print(t)

timer = Timer()

plt.ion()#启动动画的互动模式
#12.5.3 reading the dataset
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')


def load_array(data_arrays, batch_size, is_train=True):
    # 把 (X,y) 包装成标准数据集
    dataset = TensorDataset(*data_arrays)

    # 有放回抽样，抽取和数据集一样多的样本
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=len(dataset)
    )

    # 关键：用了sampler，就不能写shuffle
    data_iter = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return data_iter

def get_data_ch11_with_replacement(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))#标准化后转成张量，注意这里用了广播机制
    data_iter = load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)#加载X,y数据
    return data_iter, data.shape[1]-1#去掉了y列，所以要减一


#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))#标准化后转成张量，注意这里用了广播机制
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)#加载X,y数据
    return data_iter, data.shape[1]-1#去掉了y列，所以要减一

def sgd(params, states, hyperparams):#params=[w,b],states=none,后面章节是vt,hyperparams={'lr':lr}

    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)#只改数值，不改梯度，所以用p.data, sub_()表示原地减法.更新权重和b的值
        p.grad.data.zero_()#梯度置0

def reduce_lr(hyperparams):
    print('原学习率:',hyperparams['lr'])
    hyperparams['lr']=hyperparams['lr']*0.1
    print('修改后学习率:', hyperparams['lr'])
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    #d2l.linreg(X,w,b) =return d2l.matmul(X, w) + b
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()#只计算梯度，不更新
            trainer_fn([w, b], states, hyperparams)#更新梯度
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                #animator.add(x, y),x = 训练到第几个 epoch,y = 当前的损失值（必须放元组里）
                #X.shape[0]就是批次内的数据条数,len(data_iter)是总批次数
                #evaluate_loss会遍历总损失
                plt.pause(0.25)
                timer.start()
        reduce_lr(hyperparams)
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]

def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11_with_replacement(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(0.05, 10)
d2l.set_figsize([6, 3])
sgd_res = train_sgd(0.005, 1)
mini1_res = train_sgd(.4, 100)
mini2_res = train_sgd(.05, 10)
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
plt.ioff()
plt.show()