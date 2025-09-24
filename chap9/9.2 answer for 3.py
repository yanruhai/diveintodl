import collections
import random
import re
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt




class TimeMachine(d2l.DataModule): #@save
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download("https://archive.org/download/prideandprejudice030179gut/03017-0.txt", self.root)
        #傲慢与偏见
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]

@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()#将所有不是英文字母的字符，+表示一个或多个，替换成空格

text = data._preprocess(raw_text)
text[:60]

@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

#tokens = text.split()
tokens=data._tokenize(text)
','.join(tokens[:30])

class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):#这个if比较tokens必须不为空且内容不为空(长度>0)
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)#x是counter.items()函数的迭代后的成员
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        '''等价代码如下
        # 1. 准备各部分 token 列表
unk_token = ['<unk>']  # 未知标记
reserved = reserved_tokens  # 预留标记
filtered_tokens = [token for (token, freq) in self.token_freqs if freq >= min_freq]  # 筛选后的高频 token

# 2. 合并所有 token 列表
all_tokens = unk_token + reserved + filtered_tokens

# 3. 去重（转为集合）
unique_tokens = set(all_tokens)

# 4. 排序并转换为列表
sorted_tokens = sorted(unique_tokens)
self.idx_to_token = list(sorted_tokens)
'''

        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        #生成字典对象

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

freq_len={}
for min_f in range(0,1000,10):
    vocab = Vocab(tokens,min_freq=min_f)
    len_v=len(vocab)
    freq_len[min_f]=len_v
    print(f'最小频率为{min_f}时长度为:{len_v}')
    indices = vocab[tokens[:10]]

    print('indices:', indices)
    print('words:', vocab.to_tokens(indices))

freq_list=freq_len.keys()
vo_list=freq_len.values()
plt.figure(figsize=(8, 4))
plt.plot(freq_list,vo_list)
plt.xlabel('min_freq')
plt.ylabel('size of voc')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)

words = text.split()
#split() 方法在默认情况下（不传递参数时）会以任意数量的空白字符（包括空格、制表 符 \t、换行符 \n 等）
# 作为分隔符，并且会自动忽略字符串开头和结尾的空白。
vocab = Vocab(words)
print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
plt.show()

bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
plt.show()