import os

import numpy as np
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


class MTFraEng(d2l.DataModule):  #@save
    """The English-French dataset."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root,
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()



@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')#\u202f是窄不换行空格,\xa0是通用换行
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)#返回一个串



@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):#语料库里的句子是按行分割，然后每行用制表符分成两个句子，分别是两个语言
        if max_examples and i > max_examples: break
        parts = line.split('\t')#制表符分离
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])#eos是每个token的结束标记,if t是跳过空token,相当于len(t)>0
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt#src,tgt都是二维数组，(词汇量*分词后的长度)



data = MTFraEng()
raw_text = data._download()
print(raw_text[:75])
text = data._preprocess(raw_text)
print(text[:80])
src, tgt = data._tokenize(text)#src,tgt都是长度为167130的list,每个元素又是一个为内容是字符串的列表
print(src[:6], tgt[:6])
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    m=[[len(l) for l in xlist], [len(l) for l in ylist]]
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])#计算每个子串的长度
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)


show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt)
#plt.show()

@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())

@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    # 内部辅助函数：构建单个语言的张量和词汇表
    def _build_array(sentences, vocab, is_tgt=False):#sentences是二维列表
        pad_or_trim = lambda seq, t: (#t是标准长度，如果数组不足t就补齐，这里传入的t是num_steps,num_steps在构造方法中默认是9
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))#三元运算,如果不足长度t就补t-len(seq)个<pad>
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]#sentences执行后结果是641*9的二维数组
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = torch.tensor([vocab[s] for s in sentences])#根据单词表得到字符串的下标，生成张量
        valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)#!=符号右侧做了广播,,sum(1)得到y轴上的和,array维度为641*10
        return array, vocab, valid_len#array是单词在vocab下的下标里列表，valid_len为每个sentence的去掉<pad>的有效长度

    src, tgt = self._tokenize(self._preprocess(raw_text), self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)#src_array.shape=(641,9) src_valid_len=(641,)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)#为了后序的decoder，src值
#tgt_array[:,:-1]：解码器的输入序列，包含 <bos> 和目标序列的前 num_steps-1 个令牌，用于训练时的教师强制。
#tgt_array[:,1:]：解码器的目标输出序列，用于计算损失，包含目标序列的后续令牌（从第二个令牌到 <eos> 或 <pad>）

@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)

data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', src.type(torch.int32))
print('decoder input:', tgt.type(torch.int32))
print('source len excluding pad:', src_valid_len.type(torch.int32))
print('label:', label.type(torch.int32))

@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays

src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(src[0].type(torch.int32)))
print('target:', data.tgt_vocab.to_tokens(tgt[0].type(torch.int32)))