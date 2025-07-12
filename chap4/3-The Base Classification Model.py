import torch
from d2l import torch as d2l

class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])# * 操作符将 batch[:-1]（一个可迭代对象）解包成单独的参数
        #self 是当前模型实例，调用 self() 等价于执行模型__call__方法内的 forward() 方法。
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)#-1表示最后一列
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
        ##batch内容说明，内部结构 假设数据集是 3 个样本，每个样本 2 个特征（输入 x 是 2 维），标签 y 是 1 维
        #x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 形状 (3, 2) → 2 维
        #y = torch.tensor([0, 1, 0])  # 形状 (3,) → 1 维
        #batch = (x, y)

@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)

print('4.3.2. Accuracy')

@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))#使用 -1 自动计算第一维，Y_hat.shape[-1] 取最后一维（类别数）
    preds = Y_hat.argmax(axis=1).type(Y.dtype)
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    return compare.mean() if averaged else compare