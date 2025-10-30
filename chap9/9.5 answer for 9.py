import time
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class RNNScratch(d2l.Module):  # @save
    """The RNN model implemented from scratch."""

    def __init__(self, num_inputs, num_hiddens, sigma=1):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = nn.Parameter(
            torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))


@d2l.add_to_class(RNNScratch)  # @save
def forward(self, inputs, state=None):  # input(32*1024*28)
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        state = torch.zeros((inputs.shape[1], self.num_hiddens),
                            device=inputs.device)
    else:
        state, = state  # 等价于state = state[0] Python 中的一种解构赋值（unpacking）语法
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
        # 按照第一维循环，每次读取 (batch_size,num_inputs).依次计算h1,h2...
        state = torch.tanh(torch.matmul(X, self.W_xh) +
                           torch.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)  # 产生[h1,h2,...,ht]列表，hi∈(batch_size, num_hiddens)
    return outputs, state  # 继续新的批次可能需要当前的ht作为新批次的h0


batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = torch.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)


# print(outputs,state)

def check_len(a, n):  # @save
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'


def check_shape(a, shape):  # @save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
        f'tensor\'s shape {a.shape} != expected shape {shape}'
    # 当条件表达式为 True 时，assert 什么也不做，程序继续执行。


# 当条件表达式为 False 时，assert 会触发 AssertionError 异常，并将逗号后的 “错误提示信息” 作为异常的描述内容。

check_len(outputs, num_steps)  # 输出的长度应该等于输入的第一维,就是forward中的循环次数
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))


class RNNLMScratch(d2l.Classifier):  # @save
    """The RNN-based language model implemented from scratch."""

    def __init__(self, rnn, vocab_size, lr=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)  # 用于输出的Whq
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def training_step(self, batch):
        # batch是list[2]，如果传入batch[0]会导致为[x]数据，格式错误，所以需要解包
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=False)


@d2l.add_to_class(RNNLMScratch)  # @save
def one_hot(self, X):
    # Output shape: (num_steps, batch_size, vocab_size)
    return F.one_hot(X.T, self.vocab_size).type(torch.float32)


@d2l.add_to_class(RNNLMScratch)  # @save
def output_layer(self, rnn_outputs):
    '''将数据经过与矩阵Whq相乘得到输出'''
    # 得到维度为batch_size*one_hot张量的列表，列表按时间步排列.rnn_outputs是[h1,h2,...,ht]列表,hi∈(batch_size, num_hiddens)
    outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return torch.stack(outputs, 1)  # 将时间步放在第一维(序数是2)，堆起来，得到(batch_size*time_steps*one_hot)张量


@d2l.add_to_class(RNNLMScratch)  # @save
def forward(self, X, state=None):
    embs = self.one_hot(X)  # one_hot后得到维度(num_steps, batch_size, vocab_size)张量,在one_hot中会对X转置
    rnn_outputs, _ = self.rnn(embs, state)  # rnn_outputs是[h1,h2,...,ht]列表,hi∈(batch_size, num_hiddens)
    return self.output_layer(rnn_outputs)


start_time = time.time()  # 单位是秒
model = RNNLMScratch(rnn, num_inputs)
outputs = model(torch.ones((batch_size, num_steps), dtype=torch.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))


@d2l.add_to_class(d2l.Trainer)  # @save
def clip_gradients(self, grad_clip_val, model):  # 梯度裁剪
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))  # g的二阶范数
    print(f'Grad norm: {norm.item():.2e}')
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm


data = d2l.TimeMachine(batch_size=256, num_steps=128)  # 数据，张量为1024*32,32是时间步
rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)  # rnn中间层
model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=5)  # 模型
trainer = d2l.Trainer(max_epochs=280, gradient_clip_val=1e10, num_gpus=1)  # 训练
trainer.fit(model, data)
end_time = time.time()
print('用时：', end_time - start_time)  # 28秒
plt.show()


@d2l.add_to_class(RNNLMScratch)  # @save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]  # 获得第一个字符的字典序，用于预测
    for i in range(len(prefix) + num_preds - 1):
        X = torch.tensor([[outputs[-1]]], device=device)
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)  # 用已经训练好的权重矩阵W产生H矩阵
        if i < len(prefix) - 1:  # Warm-up period (this means prediction is in given)
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(Y.argmax(axis=2).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


print(model.predict('it has', 20, data.vocab, d2l.try_gpu()))
