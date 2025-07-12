import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from d2l import torch as d2l

# 设置随机种子确保实验可重复
torch.manual_seed(42)


# 框架实现的两层MLP
class FrameworkMLP(d2l.Classifier):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, num_hidden1),
            nn.ReLU(),
            nn.Linear(num_hidden1, num_hidden2),
            nn.ReLU(),
            nn.Linear(num_hidden2, num_outputs)
        )

    def forward(self, X):
        return self.net(X)


# 底层实现的两层MLP
class ScratchMLP(d2l.Classifier):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        # 初始化权重和偏置
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hidden1) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hidden1))
        self.W2 = nn.Parameter(torch.randn(num_hidden1, num_hidden2) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_hidden2))
        self.W3 = nn.Parameter(torch.randn(num_hidden2, num_outputs) * sigma)
        self.b3 = nn.Parameter(torch.zeros(num_outputs))

    def relu(self, X):
        return torch.max(X, torch.zeros_like(X))

    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        H1 = self.relu(torch.matmul(X, self.W1) + self.b1)
        H2 = self.relu(torch.matmul(H1, self.W2) + self.b2)
        return torch.matmul(H2, self.W3) + self.b3


# 训练和计时函数
def train_model(model, data, max_epochs):
    """训练模型并返回训练时间"""
    trainer = d2l.Trainer(max_epochs=max_epochs)
    start_time = time.time()
    trainer.fit(model, data)
    return time.time() - start_time


# 性能测试函数
def run_performance_test(data, input_size, output_size, hidden_sizes, max_epochs=10):
    """
    测试不同隐藏层大小下，框架实现和底层实现的性能差异

    参数:
        data: 数据集
        input_size: 输入维度
        output_size: 输出维度
        hidden_sizes: 隐藏层大小列表
        max_epochs: 训练轮数
    """
    framework_times = []
    scratch_times = []

    for hidden_size in hidden_sizes:
        # 框架实现
        framework_model = FrameworkMLP(
            num_inputs=input_size,
            num_hidden1=hidden_size,
            num_hidden2=hidden_size,
            num_outputs=output_size,
            lr=0.1
        )
        framework_time = train_model(framework_model, data, max_epochs)
        framework_times.append(framework_time)
        print(f"框架实现 (隐藏层大小={hidden_size}): {framework_time:.2f}秒")

        # 底层实现
        scratch_model = ScratchMLP(
            num_inputs=input_size,
            num_hidden1=hidden_size,
            num_hidden2=hidden_size,
            num_outputs=output_size,
            lr=0.1
        )
        scratch_time = train_model(scratch_model, data, max_epochs)
        scratch_times.append(scratch_time)
        print(f"底层实现 (隐藏层大小={hidden_size}): {scratch_time:.2f}秒")

    return framework_times, scratch_times


# 绘制结果
def plot_results(hidden_sizes, framework_times, scratch_times):
    """绘制性能对比图"""
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_sizes, framework_times, 'o-', label='框架实现')
    plt.plot(hidden_sizes, scratch_times, 's-', label='底层实现')
    plt.xlabel('隐藏层大小')
    plt.ylabel('训练时间 (秒)')
    plt.title('两层MLP: 框架实现 vs 底层实现')
    plt.legend()
    plt.grid(True)
    plt.savefig('mlp_performance_comparison.png')
    plt.show()


# 主函数
def main():
    # 使用指定的数据源
    data = d2l.FashionMNIST(resize=(32, 32), batch_size=256)
    input_size = 32 * 32  # 32x32图像
    output_size = 10  # Fashion-MNIST类别数

    # 测试不同复杂度
    hidden_sizes = [64, 128, 256, 512, 1024]

    print("开始性能测试...")
    framework_times, scratch_times = run_performance_test(
        data, input_size, output_size, hidden_sizes
    )

    # 计算加速比
    speedups = [s / f for s, f in zip(scratch_times, framework_times)]
    print("\n加速比:")
    for hs, su in zip(hidden_sizes, speedups):
        print(f"隐藏层大小={hs}: 框架实现比底层实现快 {su:.2f}倍")

    # 绘制结果
    plot_results(hidden_sizes, framework_times, scratch_times)


if __name__ == "__main__":
    main()