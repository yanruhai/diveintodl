import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):#lr是学习率参数
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)#第1,2个参数是均值和方差，第三个参数是张量,
        self.b = torch.zeros(1, requires_grad=True)

@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return torch.matmul(X, self.w) + self.b

print('3.4.2. Defining the Loss Function')

@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return l.mean()

print('3.4.3. Defining the Optimization Algorithm')
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):#所有的params成员都有grad字段
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):#预处理，在执行反向传播算法时需先置0
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)

print('3.4.4. Training')

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()#进入训练模式
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))#预处理 batch 数据，通常将数据移动到正确的设备（CPU/GPU）或调整格式（例如张量转换）
        self.optim.zero_grad()#将优化器（self.optim）管理的模型参数的梯度清零。
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()#更新模型参数
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1


@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l=(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()
    return l

model = LinearRegressionScratch(2, lr=0.03,sigma=0.01)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=15)
trainer.fit(model, data)
w=model.w
e1=abs(torch.sum(torch.tensor([2,-3.4])-w).detach().item())
print("e1=",e1)
b=model.b
print(abs(4.2-b))
plt.show()