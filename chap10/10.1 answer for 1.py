import multiprocessing
import os
import random
import sys
import time
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class LSTMScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input node

@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Initial state with shape: (batch_size, num_hiddens)
        H = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
        C = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
    else:
        H, C = H_C
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, self.W_xi) +
                        torch.matmul(H, self.W_hi) + self.b_i)#计算It=sigmoid(XtWxi+Ht-1Whi+bi)
        F = torch.sigmoid(torch.matmul(X, self.W_xf) +
                        torch.matmul(H, self.W_hf) + self.b_f)#计算Ft=sigmoid(XtWxf+Ht-1Whf+bf)
        O = torch.sigmoid(torch.matmul(X, self.W_xo) +
                        torch.matmul(H, self.W_ho) + self.b_o)#计算Ot=sigmoid(XtWxo+Ht-1Who+bo)
        C_tilde = torch.tanh(torch.matmul(X, self.W_xc) +
                           torch.matmul(H, self.W_hc) + self.b_c)#计算~C=tanh(XtWxc+Ht-1Whc+bc)
        C = F * C + I * C_tilde#Ct=Ft*Ct-1+It*C_tilde
        H = O * torch.tanh(C)#Ht=Ot*tanh(Ct)
        outputs.append(H)#将所有输出拼接成一个列表
    return outputs, (H, C)

class MinPerplexity:

    def start_new_stage(self,name):
        self.name=name
        self.__init__()


    def __init__(self):
        self.min_train_loss=sys.float_info.max
        self.min_val_loss=sys.float_info.max
        self.min_train_name = ''
        self.min_val_name=''
        self.all_min_train_loss = sys.float_info.max
        self.all_min_val_loss = sys.float_info.max

    def save_min(self,cur_loss,train):
        if train:
            if cur_loss<self.min_train_loss:
                self.min_train_loss=cur_loss
                if cur_loss<self.all_min_train_loss:
                    self.min_train_loss=cur_loss
                    self.min_train_name=self.name
        else:
            if cur_loss<self.min_val_loss:
                self.min_val_loss=cur_loss
                if cur_loss < self.all_min_val_loss:
                    self.min_val_loss = cur_loss
                    self.min_val_name = self.name


    def print_train_val(self):
        print(self.name,'最小训练损失:',self.min_train_loss,'最小验证损失:',self.min_val_loss)

    def print_all(self):
        print(f"在{self.min_train_name}下全局最小训练损失:{self.all_min_train_loss},在{self.min_val_name}下全局最小验证损失：{self.all_min_val_loss}")




@d2l.add_to_class(d2l.RNNLMScratch)
def __init__(self, rnn, vocab_size, lr=0.01,mp=None):
    self.__init__(rnn,vocab_size,lr)
    self.mp=mp

@d2l.add_to_class(d2l.RNNLMScratch)
def training_step(self, batch):
    l = self.loss(self(*batch[:-1]), batch[-1])
    self.plot('ppl', d2l.exp(l), train=True)
    self.save_min(l.detach().item(),True)
    return l

@d2l.add_to_class(d2l.RNNLMScratch)
def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
        self.save_min(l.detach().item(),False)




#mp.print_all()
                    #plt.show()

def train_model(params):
    print(f"进程 PID: {os.getpid()}, CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前设备: {torch.cuda.current_device()}, 内存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    else:
        print("警告: CUDA 不可用，fallback CPU!")
    # ... 其余代码 ...
    """单个超参数组合的训练函数（网格搜索任务）"""
    batch_size,num_hiddens,num_steps,lr,max_epochs = params
    mp = MinPerplexity()
    mp.start_new_stage(
        f"batch_size={batch_size},num_hiddens={num_hiddens},num_steps={num_steps},lr={lr},max_epochs={max_epochs}")
    data = d2l.TimeMachine(batch_size=batch_size, num_steps=num_steps)  # 数据
    lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=num_hiddens)  # rnn对象
    model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=lr,mp=mp)  # model对象
    trainer = d2l.Trainer(max_epochs=max_epochs, gradient_clip_val=1, num_gpus=1)  # train对象
    trainer.fit(model, data)

    #mp.print_train_val()
    return {'train_loss':mp.min_train_loss, 'val_loss':mp.min_val_loss,'mp':mp}


def cleanup(best_params, all_scores):
    """所有进程结束后执行的函数"""
    print(f"Grid search complete! Best params: {best_params}, Best score: {max(all_scores):.3f}")
    # 这里可以加保存模型、绘图等代码

def find_best_score(results,key):
    best_idx=0
    best_score=results[0][key]#暂时取0作为最优值
    for i in range(1,len(results)):
        if results[i][key]<best_score:
            best_score=results[i][key]
            best_idx=i
    return [best_idx,best_score]

if __name__ == '__main__':
    # 定义网格参数
    batch_size_list = [512]
    num_hiddens_list = [32, 64]
    num_steps_list = [32]
    lr_list = [2, 3, ]
    max_epochs_list = [ 100]
    grid_params = [(batch_size,num_hiddens,num_steps,lr,max_epochs)
                for batch_size in batch_size_list for num_hiddens in num_hiddens_list for num_steps in num_steps_list
                   for lr in lr_list for max_epochs in max_epochs_list]  # 展开成9个任务

    # 创建进程池（进程数设为CPU核心数的一半，避免过度并行）
    num_processes = 8#multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=num_processes)

    # 并行执行网格搜索
    #results = pool.map(train_model, grid_params)
    # 添加进度条（tqdm 会显示整体进度）
    with tqdm(total=len(grid_params), desc="Grid Search Progress") as pbar:
        results = []
        for res in pool.imap(train_model, grid_params):  # 用 imap 流式（见方式2优化）
            results.append(res)
            pbar.update(1)
    # 等待所有进程结束
    pool.close()
    pool.join()

    # 提取结果，找到最佳
    best_train_loss = find_best_score(results,'train_loss')
    best_val_loss= find_best_score(results,'val_loss')

    print(f'最佳的训练误差参数：{grid_params[best_train_loss[0]]}')
    print(f'最佳的验证误差参数：{grid_params[best_val_loss[0]]}')
    # 结束后调用清理函数
    #cleanup(best_params, all_scores)

    batch_size_list = [512, 1024, 2048]
    num_hiddens_list = [32, 64, 128, 256]
    num_steps_list = [32, 64, 128]
    lr_list = [2, 3, 4, 5]
    max_epochs_list = [50, 100]
    '''for batch_size in batch_size_list:
        for num_hiddens in num_hiddens_list:
            for num_steps in num_steps_list:
                for lr in lr_list:
                    for max_epochs in max_epochs_list:
                        mp.start_new_stage(
                            f"batch_size={batch_size},num_hiddens={num_hiddens},num_steps={num_steps},lr={lr},max_epochs={max_epochs}")
                        data = d2l.TimeMachine(batch_size=batch_size, num_steps=num_steps)  # 数据
                        lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=num_hiddens)  # rnn对象
                        model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=lr)  # model对象
                        trainer = d2l.Trainer(max_epochs=max_epochs, gradient_clip_val=1, num_gpus=1)  # train对象
                        trainer.fit(model, data)  # 精简实现暂时注释
                        mp.print_train_val()'''