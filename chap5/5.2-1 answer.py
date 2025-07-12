import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

torch.set_num_threads(4)  # 假设CPU有4核
print('5.2.1. Implementation from Scratch')
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs,num_h2, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_h2) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_h2))
        self.W3 = nn.Parameter(torch.randn(num_h2,num_outputs)*sigma)
        self.b3=nn.Parameter(torch.zeros(num_outputs))

print('5.2.1.2. Model')
def relu(X):#ReLU激活函数
    a = torch.zeros_like(X)
    return torch.max(X, a)

@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)
    H2=relu(torch.matmul(H,self.W2)+self.b2)
    return torch.matmul(H2, self.W3) + self.b3


def calculate_accuracy(model, dataloader, device='cpu'):
    """计算模型在数据集上的准确度"""
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)#转换数据到设备上，这里是cpu,所以不做操作
            outputs = model(inputs)#前向
            _, predicted = torch.max(outputs, 1)  # 取概率最大的类别,1表示轴方向，这里是行
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def main():
    print('5.2.1.3. Training')
   # model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
    data = d2l.FashionMNIST(resize=(32,32),batch_size=256)
    trainer = d2l.Trainer(max_epochs=10)
    #trainer.fit(model, data)
    hide_list=[2,4,8,16,32,64,128,256,512,1024]
    train_loss_list=[]
    val_loss_list=[]
    for ht in hide_list:
        model = MLPScratch(num_inputs=1024, num_outputs=10, num_h2=16,num_hiddens=ht,lr=0.1)
        trainer.fit(model, data)
        model.train()
        train_loss = calculate_accuracy(model, data.get_dataloader(train=True))
        model.eval()
        val_loss=calculate_accuracy(model,data.get_dataloader(train=False))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)


    plt.figure(figsize=(10,6))
    plt.plot(hide_list,train_loss_list,label="train_acc")
    plt.plot(hide_list,val_loss_list,label="validate_acc", linewidth=2, linestyle='--')
    plt.legend()
    plt.show()

class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))

if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()



