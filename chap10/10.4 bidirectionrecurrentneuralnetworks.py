import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

#这个例子代码有问题，跑不起来
class BiRNNScratch(d2l.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.lr=2
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.f_rnn = d2l.RNNScratch(embed_size, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(embed_size, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
        # 3. 输出层：将隐藏状态映射到词汇表维度（关键缺失部分）
        self.output = nn.Linear(self.num_hiddens, vocab_size)
        # 4. 损失函数（交叉熵损失，适用于分类任务）
        self.loss = nn.CrossEntropyLoss()

@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)
    inputs = self.embedding(inputs.T).permute(1, 0, 2)  # 适配RNN输入格式
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)#reversed() 不会修改原序列，而是返回一个新的反向迭代器
    outputs = [torch.cat((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]
    return outputs, (f_H, b_H)

@d2l.add_to_class(BiRNNScratch)
def training_step(self, batch):
    l = self.loss(self(*batch[:-1]), batch[-1])
    self.plot('ppl', d2l.exp(l), train=True)

    return l

@d2l.add_to_class(BiRNNScratch)
def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)

@d2l.add_to_class(BiRNNScratch)
def loss(self, Y_hat, Y):
    """Defined in :numref:`sec_softmax_concise`"""
    return F.cross_entropy(
        Y_hat, Y, reduction='mean' )



data = d2l.TimeMachine(batch_size=1024, num_steps=32)
vocab_size = len(data.vocab)
embed_size = 32  # 嵌入维度（可自定义，需与RNN输入维度一致）

model = BiRNNScratch(len(data.vocab),embed_size=embed_size,num_hiddens=128)
trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
plt.show()
