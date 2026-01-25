import torch
import torch.optim as optim
import numpy as np

def nadaraya_watson(x_train, y_train, x_val, kernel,sigma):
    dists = x_train.reshape((-1, 1)) - x_val.reshape((1, -1))#计算出x_train中每个点到x_val的每个点的距离
    #[[1, 1, 1],
    #[2, 2, 2],
    #[3, 3, 3]]  x_train会这样广播，x_val会沿x轴广播
    # Each column/row corresponds to each query/key
    k = kernel(dists,sigma).type(torch.float32)
    # Normalization over keys for each query
    attention_w = k / k.sum(0)#沿着第0维计算，保持第0维维度，其他维度展平
    y_hat = y_train@attention_w#矩阵乘法
    return y_hat, attention_w

def train_nadaraya_watson(x_train, y_train, x_val, y_val, sigma_init=1.0, max_epochs=500):
    # 初始化sigma
    sigma = torch.tensor(sigma_init, requires_grad=True)

    # 使用Adam优化器，它适合处理非凸优化
    optimizer = optim.Adam([sigma], lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    loss_history = []
    sigma_history = []

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # 前向传播
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, gaussian, sigma)
        loss = ((y_val - y_hat) ** 2).mean()  # 使用mean而不是sum，更稳定

        # 记录
        loss_history.append(loss.item())
        sigma_history.append(sigma.item())

        if epoch % 50 == 0:
            print(f'Epoch {epoch:3d}: loss={loss.item():.6f}, sigma={sigma.item():.6f}')

        # 反向传播和优化
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_value_([sigma], 1.0)

        optimizer.step()
        scheduler.step(loss)

        # 确保sigma为正数
        with torch.no_grad():
            sigma.data.clamp_(min=0.01, max=10.0)

        # 早停条件
        if loss < 0.05 or (epoch > 50 and abs(loss_history[-1] - loss_history[-10]) < 1e-6):
            print(f'收敛于第 {epoch} 轮，loss={loss.item():.6f}')
            break

    return sigma, loss_history, sigma_history
def f(x):
    return 2 * torch.sin(x) + x  #模拟函数 yi=2sin(xi)+xi+埃普西隆

def gaussian(x,sigma=torch.tensor(1)):
    return torch.exp(-x**2 / 2*sigma**2)

n = 40
x_train, _ = torch.sort(torch.rand(n) * 5)#torch.sort() 函数会返回一个包含两个元素的元组，即 (排序后的张量，原始元素的索引)
y_train = f(x_train) + torch.randn(n)#生成训练集
x_val = torch.arange(0, 5, 0.1)#验证集
y_val = f(x_val)#验证数据
train_nadaraya_watson(x_train, y_train, x_val,y_val,sigma_init=1.0,max_epochs=500)

