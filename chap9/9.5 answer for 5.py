import time

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = nn.Parameter(
            torch.randn(num_inputs, num_hiddens) * sigma)  # (input_dim, hidden_dim)
        self.W_hh = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * sigma)  # (hidden_dim, hidden_dim)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))    # (hidden_dim,)

    def forward(self, inputs, state=None):
        if state is None:
            # Initial state: (batch_size, num_hiddens)
            state = torch.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        else:
            state, = state  # 解构单个状态（适配d2l框架习惯）
        outputs = []
        for X in inputs:  # inputs形状: (num_steps, batch_size, num_inputs)
            # X形状: (batch_size, num_inputs)，与W_xh矩阵乘法匹配
            state = torch.tanh(torch.matmul(X, self.W_xh) +
                             torch.matmul(state, self.W_hh) + self.b_h)
            #计算Ht=XWxh+Ht-1Whh+bh
            outputs.append(state)  # 每个输出: (batch_size, num_hiddens)
        return outputs, state

# 辅助函数（保持不变）
def check_len(a, n):
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

def check_shape(a, shape):
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

# -------------------------- 核心修正1：RNNLMScratch类 --------------------------
class RNNLMScratch(d2l.Classifier):  #@save
    """修正后的RNN语言模型（可学习嵌入）"""
    # __init__参数顺序保持定义与调用一致：rnn → vocab_size → embedding_dim → lr → min_ppl
    def __init__(self, rnn, vocab_size, embedding_dim, lr=0.01, min_ppl=100):
        super().__init__()
        # 可学习嵌入层：(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.save_hyperparameters()  # 保存所有超参数（方便后续查看）
        self.init_params()
        self.changed = False

    # -------------------------- 核心修正2：输出层权重维度 --------------------------
    def init_params(self):
        # W_hq：(num_hiddens, vocab_size) → 隐藏态映射到词汇表维度
        self.W_hq = nn.Parameter(
            torch.randn(self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))  # 偏置：(vocab_size,)

    def training_step(self, batch):
        X, Y = batch  # batch是(input, label)对，直接解包更清晰
        Y_hat = self(X)  # 前向传播
        l = self.loss(Y_hat.reshape(-1, self.vocab_size), Y.reshape(-1))  # 展平后计算损失
        self.plot('ppl', torch.exp(l), train=True)  # 困惑度=exp(交叉熵损失)
        return l

    def validation_step(self, batch):
        X, Y = batch
        Y_hat = self(X)
        l = self.loss(Y_hat.reshape(-1, self.vocab_size), Y.reshape(-1))
        cur_ppl = torch.exp(l)
        if cur_ppl < self.min_ppl:
            self.min_ppl = cur_ppl
            self.changed = True
        self.plot('ppl', cur_ppl, train=False)

@d2l.add_to_class(RNNLMScratch)
def output_layer(self, rnn_outputs):
    # rnn_outputs：列表，每个元素是(batch_size, num_hiddens)
    # 每个时间步输出：(batch_size, vocab_size)
    outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    # 堆叠后形状：(batch_size, num_steps, vocab_size)（匹配训练数据Y的维度）
    return torch.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)
def forward(self, X, state=None):
    # X：(batch_size, num_steps) → 嵌入层输出：(batch_size, num_steps, embedding_dim)
    embs = self.embedding(X)
    # 调整维度为RNN输入要求：(num_steps, batch_size, embedding_dim)
    embs = embs.permute(1, 0, 2)
    # RNN前向传播：输出是列表，每个元素(batch_size, num_hiddens)
    rnn_outputs, _ = self.rnn(embs, state)
    # 输出层映射到词汇表维度
    return self.output_layer(rnn_outputs)

# -------------------------- 核心修正3：predict方法用嵌入层 --------------------------
@d2l.add_to_class(RNNLMScratch)
def predict(self, prefix, num_preds, vocab, device=None):
    state = None  # 初始状态为None
    outputs = [vocab[prefix[0]]]  # 前缀的第一个字符索引
    for i in range(len(prefix) + num_preds - 1):
        # 输入X：(1, 1) → (batch_size=1, num_steps=1)
        X = torch.tensor([[outputs[-1]]], device=device)
        # 用嵌入层（与训练一致），而非one_hot
        embs = self.embedding(X).permute(1, 0, 2)  # 调整为(1, 1, embedding_dim)
        # RNN推理：输出rnn_outputs是列表，长度1
        rnn_outputs, state = self.rnn(embs, state)
        # 预热阶段（prefix内）：直接取前缀的下一个字符
        if i < len(prefix) - 1:
            outputs.append(vocab[prefix[i + 1]])
        # 预测阶段：取概率最大的字符索引
        else:
            Y = self.output_layer(rnn_outputs)  # Y形状：(1, 1, vocab_size)
            outputs.append(int(Y.argmax(dim=2).reshape(1)))
    # 索引转字符，拼接成字符串
    return ''.join([vocab.idx_to_token[idx] for idx in outputs])

# -------------------------- 核心修正4：梯度裁剪方法（保持不变，但需确保生效） --------------------------
@d2l.add_to_class(d2l.Trainer)
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    # 计算所有参数梯度的L2范数
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > grad_clip_val:
        # 梯度缩放：确保范数不超过clip_val
        for param in params:
            param.grad[:] *= grad_clip_val / norm

# -------------------------- 测试与训练（修正参数传递和维度） --------------------------
if __name__ == "__main__":
    start_time=time.time()#单位是秒
    # 1. 加载数据集（TimeMachine：字符级语言模型）
    batch_size = 1024  # 批次大小（可调整）
    num_steps = 32    # 截断时间步（RNN输入序列长度）
    data = d2l.TimeMachine(batch_size=batch_size, num_steps=num_steps)
    vocab_size = len(data.vocab)  # 词汇表大小（字符级，约100）
    embedding_dim = 64            # 嵌入向量维度（超参数，可调整为128）
    num_hiddens = 128             # RNN隐藏层维度（超参数）
    lr = 1.0                      # 学习率（RNN常用较大学习率）
    max_epochs = 50               # 训练轮次（50轮足够看到困惑度下降）

    # 2. 创建RNN：num_inputs=embedding_dim（关键！匹配嵌入层输出维度）
    rnn = RNNScratch(num_inputs=embedding_dim, num_hiddens=num_hiddens)

    # 3. 创建模型：参数顺序与__init__一致（rnn → vocab_size → embedding_dim → lr）
    model = RNNLMScratch(
        rnn=rnn,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lr=lr
    )

    # 4. 训练：梯度裁剪值设为1（防止梯度爆炸）
    trainer = d2l.Trainer(max_epochs=max_epochs, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)
    end_time=time.time()
    print('用时：',end_time-start_time)#结果是29s
    plt.show()

    # 5. 测试推理：预测前缀"time traveler"后的100个字符
    prefix = "time traveler"
    num_preds = 100
    result = model.predict(prefix, num_preds, data.vocab)
    print(f"预测结果：{result}")