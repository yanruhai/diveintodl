import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)#这里的_符号表示原地操作，不会产生新数据
        self.net.bias.data.fill_(0)

@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)#self.net.__call__(X)  # 简化为 self.net.forward(X)

print('3.5.2. Defining the Loss Function')

@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.HuberLoss()#同样的调用__call__函数
    return fn(y_hat, y)

print('3.5.3. Defining the Optimization Algorithm')

@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)

@d2l.add_to_class(LinearRegression)  #@save
def get_gradient_weight(self):
    op=self.configure_optimizers()

    return op.state_dict()
print('3.5.4. Training')

k0=5
x_s=[]
y_s=[]
for ij in range(12):
    tk=k0*2**ij
    x_s.append(tk)
    model = LinearRegression(lr=0.03)
    data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2,num_train=tk,num_val=tk)
    trainer = d2l.Trainer(max_epochs=3)
    trainer.fit(model, data)
    yt=torch.sum(model.net.weight-torch.tensor([2, -3.4]))+4.2-model.net.bias
    y_s.append(yt.detach().item())
plt.show()
plt.plot(x_s,y_s)
plt.show()


plt.show()

@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)
w, b = model.get_w_b()




print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
t=model.get_gradient_weight()
print(model.get_gradient_weight())
print()

