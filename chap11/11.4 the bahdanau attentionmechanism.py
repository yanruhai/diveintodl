import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
import time

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
#以下是从库中取出的代码
class AdditiveAttention(nn.Module):
    """Additive attention.

    Defined in :numref:`subsec_batch_dot`"""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        #queries :解码器当前步最后一层的隐藏状态,(128,1,256)
        #keys:编码器所有时间步的输出（enc_outputs）(128,9,256),即(batch_size,num_steps,num_hiddens)
        #values=keys
        queries, keys = self.W_q(queries), self.W_k(keys)#运算后维度不变
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)#这里增加维度，为了防止queries里第二个维度不是1
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)#features(128,1,9,256)经过w_v后变成(128,1,9,1),squeeze去掉最后一维，128,1,9
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        tt=torch.bmm(self.dropout(self.attention_weights), values)#补充说明，最后只剩(batch_size,hiddens)
        return torch.bmm(self.dropout(self.attention_weights), values)#权重与v相乘，注意力的最后一步

class AttentionDecoder(d2l.Decoder):  #@save
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):#state源于encoder的
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).encoder的隐藏层H最后一层状态列表
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens) encoder的最后一个状态，可能是多层
        enc_outputs, hidden_state, enc_valid_lens = state
        #enc_outputs就是k,v
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)#只用最后一层做注意力的Query,
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)#context=(128,1,256)就是(batch_size,注意力结果,num_hiddens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)#运行后x=128,1,512,context=128,1,256
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)#rnn的前向获得当前时间步的token特征,并更新hidden_state,此时多层状态传入
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)#self.attention.attention_weights=(128,1,9),将新的注意力加入
            #128是batch_size,9是num_steps，这个注意力用于可视化
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))#从9,128,256转成9,128,214
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

#下面是d2l库中的EncoderDecoder类,仅仅作为说明
class EncoderDecoder(d2l.Classifier):
    """The base class for the encoder--decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        ##enc_all_outputs包括outputs:H,state=(hidden_state,cell_state)
        # args是enc_valid_lens
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        # return (outputs.permute(1, 0, 2), state, enc_valid_lens)
        return self.decoder(dec_X, dec_state)[0]

    def predict_step(self, batch, device, num_steps,
                     save_attention_weights=False):
        """Defined in :numref:`sec_seq2seq_training`"""
        batch = [d2l.to(a, device) for a in batch]
        src, tgt, src_valid_len, _ = batch#src, tgt, src_valid_len, tgt_valid_len,src=(4,9)第一维是批次，第二维是时间步
        enc_all_outputs = self.encoder(src, src_valid_len)#走一次encoder,前向一次性计算获得源src上下文
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)#转换decoder的隐参数，仅仅是数据格式转换
        outputs, attention_weights = [d2l.expand_dims(tgt[:, 0], 1), ], []#从目标 batch (tgt) 中取出所有句子的第 0 个特征
        #expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)#-1表示上一步生成的，就是自回归
            #Y=(4,1,214)
            outputs.append(d2l.argmax(Y, 2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return d2l.concat(outputs[1:], 1), attention_weights


'''vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7
encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens,
                                  num_layers)
X = torch.zeros((batch_size, num_steps), dtype=torch.long)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
d2l.check_shape(output, (batch_size, num_steps, vocab_size))
d2l.check_shape(state[0], (batch_size, num_steps, num_hiddens))
d2l.check_shape(state[1][0], (batch_size, num_hiddens))'''

start=time.time()
data = d2l.MTFraEng(batch_size=128)#默认时间步是9
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
encoder = d2l.Seq2SeqEncoder(
    len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],lr=0.005)
trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
end=time.time()
# 或者更漂亮一点：
print(f"运行耗时：{(end - start)*1000:.1f} 毫秒")
plt.show()

engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
preds, _ = model.predict_step(
    data.build(engs, fras), d2l.try_gpu(), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')


_, dec_attention_weights = model.predict_step(
    data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
#dec_attention_weights是长度为9的列表，每个元素是长度为1的列表，列表内是1，1，9的张量，因此下面代码中step[0][0][0]取出的维度是(9,)
#注意力权重的标准维度通常是：(Batch Size,Num Heads,Query Length, Key Length)
attention_weights = torch.cat(
    [step[0][0][0] for step in dec_attention_weights], 0)
attention_weights = attention_weights.reshape((1, 1, -1, data.num_steps))
#转成1,1,9,9维度

# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
#只显示最后一个元素的热力图




