import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l


# 批量归一化实现（保持不变）
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y


class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.kernels = {}
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(num_classes))

    def save_kernels(self, save_dir='kernels'):
        os.makedirs(save_dir, exist_ok=True)
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Conv2d):
                with torch.no_grad():
                    kernel = layer.weight.detach().cpu().numpy()  # 形状 (out_channels, in_channels, H, W)
                    # 保存为 .npy 文件
                    np.save(f'{save_dir}/kernel_layer_{i}.npy', kernel)
                    print(f'Saved kernel for layer {i} with shape {kernel.shape}')
                    self.kernels[i] = kernel

                    # 保存为图片（每个过滤器一张图）
                    os.makedirs(f'{save_dir}/kernel_images_layer_{i}', exist_ok=True)
                    out_channels, in_channels = kernel.shape[0], kernel.shape[1]
                    for out_ch in range(out_channels):
                        for in_ch in range(in_channels):
                            filter = kernel[out_ch, in_ch]  # 形状 (H, W)
                            # 归一化到 [0, 1] 以便可视化
                            filter = (filter - filter.min()) / (filter.max() - filter.min() + 1e-8)
                            plt.imsave(
                                f'{save_dir}/kernel_images_layer_{i}/filter_{out_ch}_{in_ch}.png',
                                filter, cmap='gray'
                            )
                            print(
                                f'Saved kernel image for layer {i}, filter {out_ch}_{in_ch} with shape {filter.shape}')
        torch.cuda.empty_cache()

    def forward(self, X):
        return self.net(X)


def main():
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128)
    model = BNLeNet(lr=0.01)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model, data)

    # 保存卷积核（.npy 和图片）
    model.eval()
    model.save_kernels()
    plt.show()


if __name__ == "__main__":
    main()