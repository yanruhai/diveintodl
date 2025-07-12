import numpy as np
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

print('4.4.1. The Softmax')
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdims=True), X.sum(1, keepdims=True))


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)  # 若原始维度：[m, n]，则求和后维度：[m, 1]
    return X_exp / partition  # The broadcasting mechanism is applied here


X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))

print('4.4.2. The Model')


class SoftmaxRegressionScratch(d2l.Classifier):  # Module的子类
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)  # 先随机对数据初始化，等待后序更新
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]


@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):  # 批量处理batch,注意x已经被处理为batch*784的矩阵
    X = X.reshape((-1, self.W.shape[0]))  # 即使预处理正确，若用户手动输入形状异常的数据（如[784, 1]或[1, 1, 784]），reshape能自动修正。
    return softmax(torch.matmul(X, self.W) + self.b)


print('4.4.3. The Cross-Entropy Loss')

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])


# print(y_hat[[0, 1], y])

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()  # len(y_hat)得到第一维的长度，即2


print(cross_entropy(y_hat, y))


@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)


print('4.4.4. Training')


def main():
    # 初始化数据、模型和训练器
    lr_list = np.arange(0.05, 0.9, step=0.05)
    loss_list=[]
    for lr in lr_list:
        data = d2l.FashionMNIST(batch_size=256)
        model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
        trainer = d2l.Trainer(max_epochs=10)

        # 训练模型
        trainer.fit(model, data)

        X, y = next(iter(data.val_dataloader()))
        preds = model(X).argmax(axis=1)  # 预测,preds是最大值下标
        diff = (preds != y).int()  # True -> 1, False -> 0

        sum_diff = diff.sum()
        loss_list.append(sum_diff)
    plt.figure()
    plt.plot(lr_list,loss_list)
    plt.show()


if __name__ == "__main__":  # 确保代码块只在直接运行时执行，而在被导入时不执行
    main()
