import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义无批量归一化的LeNet模型
class LeNetNoBN(nn.Module):
    def __init__(self):
        super(LeNetNoBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)#通过 view 方法重塑张量，将从第 1 维开始的所有维度合并为一个维度，结果形状为 (batch_size, 特征总数)。\
        #x.size(0)==x.shape(0)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义带有批量归一化的LeNet模型
class LeNetWithBN(nn.Module):
    def __init__(self):
        super(LeNetWithBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    训练模型并记录训练过程中的准确率和损失

    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备（cpu或cuda）

    返回:
        训练和测试的准确率、损失记录
    """
    # 初始化记录列表
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # 将模型移动到指定设备
    model.to(device)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和参数更新
            loss.backward()
            optimizer.step()#更新参数

            # 记录损失，loss.item() 用于将张量转换为 Python 浮点数，方便数值计算和累积。
            train_loss += loss.item() * inputs.size(0)

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)#获得维度1上的最大值，第0维是批次,结果是向量[64,1]
            total_train += labels.size(0)#累计总的训练量
            correct_train += (predicted == labels).sum().item()

        # 计算平均训练损失和准确率
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():  # 关闭梯度计算
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # 计算平均测试损失和准确率
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_acc = correct_test / total_test

        # 保存结果
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_acc)

        # 打印 epoch 结果
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f}')
        print('-' * 50)

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }


# 绘制结果对比图
def plot_results(results_no_bn, results_with_bn, num_epochs):
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    """绘制带有和不带有批量归一化的模型结果对比图"""
    epochs = range(1, num_epochs + 1)

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制准确率对比
    ax1.plot(epochs, results_no_bn['test_accuracies'], 'b-', label='无批量归一化')
    ax1.plot(epochs, results_with_bn['test_accuracies'], 'r-', label='有批量归一化')
    ax1.set_title('测试准确率对比')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True)

    # 绘制损失对比
    ax2.plot(epochs, results_no_bn['test_losses'], 'b-', label='无批量归一化')
    ax2.plot(epochs, results_with_bn['test_losses'], 'r-', label='有批量归一化')
    ax2.set_title('测试损失对比')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 定义超参数
    num_epochs = 10
    learning_rate = 0.01

    # 初始化模型
    model_no_bn = LeNetNoBN()
    model_with_bn = LeNetWithBN()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()#parameters()只会返回所有的可学习的超参数
    optimizer_no_bn = optim.SGD(model_no_bn.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_with_bn = optim.SGD(model_with_bn.parameters(), lr=learning_rate, momentum=0.9)

    # 训练无批量归一化的模型
    print("开始训练无批量归一化的LeNet模型...")
    results_no_bn = train_model(
        model_no_bn, train_loader, test_loader,
        criterion, optimizer_no_bn, num_epochs, device
    )

    # 训练有批量归一化的模型
    print("开始训练有批量归一化的LeNet模型...")
    results_with_bn = train_model(
        model_with_bn, train_loader, test_loader,
        criterion, optimizer_with_bn, num_epochs, device
    )

    # 绘制结果对比
    plot_results(results_no_bn, results_with_bn, num_epochs)

    # 测试不同学习率的影响
    test_learning_rates()


def test_learning_rates():
    """测试不同学习率对模型的影响，找到优化失效前的最大学习率"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    num_epochs = 5  # 为了节省时间，使用较少的epoch

    # 存储两种模型在不同学习率下的表现
    no_bn_results = []
    with_bn_results = []

    criterion = nn.CrossEntropyLoss()

    print("\n开始测试不同学习率的影响...")
    for lr in learning_rates:
        print(f"\n测试学习率: {lr}")

        # 无批量归一化模型
        model_no_bn = LeNetNoBN()
        optimizer_no_bn = optim.SGD(model_no_bn.parameters(), lr=lr, momentum=0.9)
        results = train_model(
            model_no_bn, train_loader, test_loader,
            criterion, optimizer_no_bn, num_epochs, device
        )
        no_bn_results.append(results['test_accuracies'][-1])  # 取最后一个epoch的准确率

        # 有批量归一化模型
        model_with_bn = LeNetWithBN()
        optimizer_with_bn = optim.SGD(model_with_bn.parameters(), lr=lr, momentum=0.9)
        results = train_model(
            model_with_bn, train_loader, test_loader,
            criterion, optimizer_with_bn, num_epochs, device
        )
        with_bn_results.append(results['test_accuracies'][-1])  # 取最后一个epoch的准确率

    # 绘制不同学习率下的准确率对比
    plt.figure(figsize=(10, 6))
    plt.semilogx(learning_rates, no_bn_results, 'bo-', label='无批量归一化')
    plt.semilogx(learning_rates, with_bn_results, 'ro-', label='有批量归一化')
    plt.title('不同学习率对模型准确率的影响')
    plt.xlabel('学习率 (log scale)')
    plt.ylabel('测试准确率')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 分析结果，找到优化失效前的最大学习率
    print("\n学习率测试结果分析:")
    print(f"无批量归一化模型在学习率为 {learning_rates[np.argmax(no_bn_results)]} 时表现最佳")
    print(f"有批量归一化模型在学习率为 {learning_rates[np.argmax(with_bn_results)]} 时表现最佳")


if __name__ == '__main__':
    main()
