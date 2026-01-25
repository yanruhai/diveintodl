import collections
import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def init_seq2seq(module):  # @save
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:  # 判断 子串 in 字符串相当于contain函数
                # nn.GRU中，对每个权重矩阵定义按照规范-
                # weight_ih_l0：第 0 层（唯一层）的 “输入→隐藏层” 权重（对应 GRU 的三个门）；
                # - weight_hh_l0：第 0 层的 “隐藏层→隐藏层” 权重；
                # - bias_ih_l0/bias_hh_l0：对应权重的偏置参数；
                nn.init.xavier_uniform_(module._parameters[param])


class Seq2SeqEncoder(d2l.Encoder):  # @save
    """The RNN encoder for sequence-to-sequence learning."""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers,
                           dropout)  # 这里的dropout仅当 num_layers > 1 时，在除最后一层之外的每一层 GRU 的输出后应用 Dropout
        self.apply(init_seq2seq)  # 将self作为参数传给init_seq2seq里面module参数

    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(X.t().type(torch.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)最后一层的隐藏状态
        # state shape: (num_layers, batch_size, num_hiddens),各层的最后一个隐藏状态
        return outputs, state


'''vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 9
encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
X = torch.zeros((batch_size, num_steps))
enc_outputs, enc_state = encoder(X)
d2l.check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))'''


class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = d2l.GRU(embed_size + num_hiddens, num_hiddens,
                           num_layers, dropout)  # 需要补充encoder中产生的c参数到运算中，所以维度有变化
        self.dense = nn.LazyLinear(vocab_size)  # 又称全连接层或稠密层
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, e
        # mbed_size)
        embs = self.embedding(X.t().type(torch.int32))  # 需要将X的batch_size和num_steps调换后生成embedding数据
        enc_output, hidden_state = state  # hidden_state是每一层的最后一个隐藏状态
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]  # 就是模型中的c向量
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)  # 先广播,获得张量维度为(num_steps, batch_size, num_hiddens)
        # Concat at the feature dimension
        embs_and_context = torch.cat((embs, context),
                                     -1)  # cat后生成维度为num_steps,batch_size,embed_size+num_hiddens，-1表示最后一个维度
        outputs, hidden_state = self.rnn(embs_and_context,
                                         hidden_state)  # outputs shape: (num_steps, batch_size, num_hiddens)
        # 此时需要传入[x,c]数组，hidden_state是encoder的每一层的最后一个隐藏状态列表
        # 传入hidden_state是因为decoder也是多层的，第一次初始化decoder的h时候要用encoder的最后一个同层的h初始化
        outputs = self.dense(outputs).swapaxes(0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size) 即使有embedding层，最后也需要将数据映射到vocab_size里
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]


'''decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
state = decoder.init_state(encoder(X))
dec_outputs, state = decoder(X, state)
d2l.check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
d2l.check_shape(state[1], (num_layers, batch_size, num_hiddens))'''


class Seq2Seq(d2l.EncoderDecoder):  # @save
    """The RNN encoder--decoder for sequence to sequence learning."""

    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, enc_X, dec_X, *args):  # 抄袭自torch.py源码
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)  # 初始化decoder的隐藏状态
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]


@d2l.add_to_class(Seq2Seq)
def loss(self, Y_hat, Y):
    l = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
    mask = (Y.reshape(-1) != self.tgt_pad).type(torch.float32)  # self.tgt_pad就是<padding>标记
    # return (l).sum()
    return (l * mask).sum() / mask.sum()  # 只记录有效平均损失


# src_vocab=194,tgt_vocat=214
data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 512, 512, 2, 0.2
encoder = Seq2SeqEncoder(
    len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(
    len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                lr=0.005)
trainer = d2l.Trainer(max_epochs=40, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)


@d2l.add_to_class(d2l.EncoderDecoder)  # @save
def predict_step(self, batch, device, num_steps,
                 save_attention_weights=False,k=3):
    batch = [a.to(device) for a in batch]  # 不改变batch的维度，[src,tgt,src_valid_len,tgt_valid_len]
    src, tgt, src_valid_len, _ = batch
    enc_all_outputs = self.encoder(src, src_valid_len)  # 忽略encoder中的前向返回值的第二个值state,预热数据
    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [tgt[:, (0)].unsqueeze(1), ], []
    Y, dec_state = self.decoder(outputs[-1], dec_state)#先处理根节点
    root_list_values,root_list_indices=torch.topk(Y,k,dim=2)#前k个节点的概率列表和对应字典的下标，得到batch_size*1*k张量

    for i in range(k):
        root=Node(root_list_values[:,:,i],root_list_indices[i],None,dec_state)
        create_tree_k_step(root,k,num_steps-1,num_steps)#parent：父节点，k：获取k个,cur_step,
        # num_steps:当前和总的时间步,des_state：用于生成预测的状态

    '''for _ in range(num_steps):
        Y, dec_state = self.decoder(outputs[-1], dec_state)
        outputs.append(Y.argmax(2))
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weights.append(self.decoder.attention_weights)'''
    return torch.cat(outputs[1:], 1), attention_weights


class Node:
    def __init__(self, p, token,parent,dec_state):
        self._sub_node = []  # 子节点列表
        self._p=p#改词的概率
        self._logp = math.log(p)  # 当前节点的概率值的log和
        self._token = token
        self._parent=parent#父节点
        self._dec_state=dec_state

    @property
    def probability(self):
        return self._p

    @property
    def dec_state(self):
        return self._dec_state

    @property
    def parent(self):
        return self._parent

    @property
    def sub_node(self):
        return self._sub_node

    def add_sub_node(self, node):
        self.sub_node.append(node)

    @property
    def log_p(self):
        return self._logp

    @log_p.setter
    def log_p(self,value):
        self._logp=value

    @property
    def token(self):
        return self._token


@d2l.add_to_class(d2l.EncoderDecoder)
def create_tree_k_step(self,parent, k, cur_step,num_steps):
    Y, dec_state = self.decoder(parent.token, parent.dec_state)
    list_values, list_indices = torch.topk(Y, k, dim=2)  # 前k个节点的概率列表和对应字典的下标
    if cur_step<num_steps and parent.token!=data.tgt_vocab['<eos>']:#没有到叶子
        for i in range(k):
            sub_node=Node(list_values[i],list_indices[i],parent,dec_state)#dec_state已更新
            sub_node.log_p=parent.log_p+math.log(sub_node.probability)
            parent.add_sub_node(sub_node)
            create_tree_k_step(self,sub_node,k,cur_step+1,num_steps)
    else:#到叶子
        for i in range(k):
            sub_node = Node(list_values[i], list_indices[i], parent, dec_state)
            sub_node.log_p = (parent.log_p + math.log(sub_node.probability))*(1/cur_step**0.75)#按照公式算出最后的结果值
            parent.add_sub_node(sub_node)








plt.show()


def bleu(pred_seq, label_seq, k):  # @save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# Ghosts exist.	Les fantômes existent.
# The army had plenty of weapons.	L'armée disposait de tas d'armes.
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
tt = data.build(engs, fras)
preds, _ = model.predict_step(
    data.build(engs, fras), d2l.try_gpu(), data.num_steps)
bleu_total = 0
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    score = bleu(" ".join(translation), fr, k=2)
    bleu_total += score
    print(f'{en} => {translation}, bleu,'
          f'{score:.3f}')
print(f'total_bleu={bleu_total}')