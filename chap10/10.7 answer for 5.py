import collections
import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l


# 1. 修正后的初始化函数
def init_seq2seq(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) in [nn.GRU, nn.LSTM]:
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)


# 2. Encoder 使用 LSTM
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers, dropout=dropout)
        self.apply(init_seq2seq)

    def forward(self, X, *args):
        embs = self.embedding(X.t().type(torch.int64))
        # state 是 (h_n, c_n) 元组
        outputs, state = self.rnn(embs)
        return outputs, state


# 3. Decoder 适配 LSTM 状态传递
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 输入维度：embedding + num_hiddens (因为拼接了上下文向量c)
        self.rnn = nn.LSTM(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        embs = self.embedding(X.t().type(torch.int64))
        enc_output, hidden_state = state

        # context 取 Encoder 最后一个时间步的输出 (也就是 c)
        context = enc_output[-1]
        context = context.repeat(embs.shape[0], 1, 1)

        # 拼接嵌入向量和上下文向量
        embs_and_context = torch.cat((embs, context), -1)

        # 这里的 hidden_state 是 (h, c) 元组，直接传给 LSTM
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)

        outputs = self.dense(outputs).swapaxes(0, 1)
        # 返回结果并保持状态结构用于循环预测
        return outputs, [enc_output, hidden_state]


# 4. 模型配置与训练
data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.1
encoder = Seq2SeqEncoder(len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=0.005)

trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)


# 5. 预测与评估
def predict_seq2seq(model, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    model.to(device)  # 确保模型整体在正确的设备上
    model.eval()

    # 1. 处理输入序列
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])

    # 2. 关键点：创建 tensor 时必须指定 device=device
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)

    # Encoder 预测
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)

    # 3. 准备 Decoder 的第一个输入 <bos>，同样要指定 device
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)

    output_seq = []
    for _ in range(num_steps):
        # Decoder 预测
        Y, dec_state = model.decoder(dec_X, dec_state)
        # 取概率最大的词作为下一个预测词
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()

        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)

    return ' '.join(tgt_vocab.to_tokens(output_seq))


# 测试
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

for en, fr in zip(engs, fras):
    translation = predict_seq2seq(model, en, data.src_vocab, data.tgt_vocab, data.num_steps, d2l.try_gpu())
    print(f'{en} => {translation}, bleu: {d2l.bleu(translation, fr, k=2):.3f}')