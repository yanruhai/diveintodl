import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing  # 替换为加州房价数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 加载并预处理数据（替换为加州房价数据集）
# 加载数据集：返回的是Bunch对象，包含data（特征）、target（目标值，即房价）
california_housing = fetch_california_housing()
X = california_housing.data  # 特征：共8个维度（如平均收入、房屋年龄、房间数等）
y = california_housing.target  # 目标值：房屋中位数价格（单位：10万美元）

# 分割训练集和测试集（测试集占比20%，固定随机种子确保结果可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化：特征标准化是神经网络训练的关键步骤，避免因特征尺度差异影响收敛
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 训练集：拟合+标准化（用训练集的均值/方差）
X_test = scaler.transform(X_test)  # 测试集：仅用训练集的均值/方差标准化（避免数据泄露）

# 转换为PyTorch张量：神经网络需用张量输入，且目标值需转为列向量（形状为[样本数, 1]）
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # view(-1,1)：自动计算行数，列数固定为1
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据集和数据加载器：TensorDataset包装特征和目标，DataLoader实现批量加载+打乱（训练集）
train_dataset = TensorDataset(X_train, y_train)  # 训练集：特征+目标
test_dataset = TensorDataset(X_test, y_test)  # 测试集：特征+目标

# 训练集：batch_size=32（每次迭代用32个样本），shuffle=True（每次epoch前打乱数据，避免过拟合）
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 测试集：shuffle=False（无需打乱，仅需评估）
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 2. 定义回归模型（三层全连接神经网络）
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()  # 继承nn.Module的初始化方法
        # 定义神经网络层：用nn.Sequential按顺序包装层（前向传播时按顺序执行）
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),  # 输入层→隐藏层1：输入维度=input_size，输出维度=64
            nn.ReLU(),  # 激活函数：ReLU（解决线性模型的表达能力不足问题）
            nn.Linear(64, 32),  # 隐藏层1→隐藏层2：输入维度=64，输出维度=32
            nn.ReLU(),  # 激活函数：ReLU
            nn.Linear(32, 1)  # 隐藏层2→输出层：输入维度=32，输出维度=1（回归任务仅需1个预测值）
        )

    # 前向传播：定义数据如何通过网络层计算预测值
    def forward(self, x):
        return self.layers(x)  # x输入后，经过self.layers的所有层，返回预测值


# 初始化模型：input_size=特征维度（加州房价数据集共8个特征）
input_size = X_train.shape[1]  # X_train.shape=(样本数, 特征数)，取第1维即特征数
model = RegressionModel(input_size)

# 3. 定义损失函数和优化器（回归任务核心配置）
criterion = nn.MSELoss()  # 损失函数：均方误差（MSE），回归任务最常用（计算预测值与真实值的平方差均值）
optimizer = optim.Adam(  # 优化器：Adam（自适应学习率优化器，比SGD收敛更快、更稳定）
    model.parameters(),  # 待优化的参数：模型中所有可训练参数（权重、偏置）
    lr=0.001  # 学习率：控制参数更新的步长（0.001是常用初始值）
)

# 4. 训练模型（核心循环：多轮迭代更新参数）
epochs = 100  # 训练轮数：整个数据集遍历100次
train_losses = []  # 记录每轮训练损失（用于后续可视化）
test_losses = []  # 记录每轮测试损失（用于后续可视化）

for epoch in range(epochs):
    # 训练模式：启用 dropout/batch norm 等训练特有的层（此处无，但养成习惯）
    model.train()
    train_loss = 0.0  # 累积当前轮的训练损失

    # 按批次加载训练数据：每次取一个batch的样本
    for batch_X, batch_y in train_loader:
        # 1. 前向传播：计算当前batch的预测值
        outputs = model(batch_X)
        # 2. 计算损失：预测值与真实值的MSE
        loss = criterion(outputs, batch_y)

        # 3. 反向传播+参数更新：
        optimizer.zero_grad()  # 清零梯度（避免上一轮梯度累积）
        loss.backward()  # 反向传播：计算各参数的梯度
        optimizer.step()  # 更新参数：根据梯度和学习率调整参数

        # 累积损失：loss.item()是当前batch的平均损失，乘以batch大小得到总损失
        train_loss += loss.item() * batch_X.size(0)

    # 计算当前轮的平均训练损失（总损失 / 训练集总样本数）
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # 测试模式：禁用 dropout/batch norm 等训练特有的层，固定模型参数
    model.eval()
    test_loss = 0.0  # 累积当前轮的测试损失

    # 禁用梯度计算：测试阶段仅需前向传播，无需反向传播（节省内存+加速）
    with torch.no_grad():
        # 按批次加载测试数据
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)  # 前向传播计算预测值
            loss = criterion(outputs, batch_y)  # 计算测试损失
            test_loss += loss.item() * batch_X.size(0)  # 累积测试损失

    # 计算当前轮的平均测试损失
    avg_test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)

    # 每10轮打印一次训练进度（方便观察收敛情况）
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

# 5. 可视化训练过程：观察训练/测试损失的下降趋势（判断是否收敛、过拟合）
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss (MSE)')  # 训练损失曲线
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss (MSE)')  # 测试损失曲线
plt.xlabel('Epochs (训练轮数)')
plt.ylabel('Loss (均方误差)')
plt.title('Training and Testing Loss Over Epochs')
plt.legend()  # 显示图例
plt.grid(alpha=0.3)  # 显示网格（方便读数）
plt.show()

# 6. 模型评估：计算测试集的RMSE（均方根误差，更直观反映房价预测误差）
model.eval()
with torch.no_grad():
    y_pred = model(X_test)  # 对整个测试集计算预测值
    mse = criterion(y_pred, y_test)  # 测试集MSE
    rmse = torch.sqrt(mse)  # RMSE = sqrt(MSE)（将误差还原到原数据尺度）
    # 加州房价目标值单位是“10万美元”，因此RMSE单位也是“10万美元”
    print(f'Test Set RMSE (均方根误差): {rmse.item():.4f} × 10万美元')

# 可视化预测结果：散点图展示“真实值 vs 预测值”（理想情况应贴近对角线）
plt.figure(figsize=(10, 6))
# 散点图：x=真实房价，y=预测房价，alpha=0.6（透明度，避免点重叠）
plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.6, label='Predicted vs Actual')
# 对角线：y=x（理想预测线，若点贴近这条线，说明预测越准确）
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Prediction (y=x)')
plt.xlabel('Actual Housing Price (× 10万美元)')
plt.ylabel('Predicted Housing Price (× 10万美元)')
plt.title('Actual vs Predicted Housing Prices (California Dataset)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()