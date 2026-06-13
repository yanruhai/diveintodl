import collections
import math
import os
import random
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt
# 1. 强制重写库函数的返回值，让它在你的脚本运行期间始终返回 0
def get_dataloader_workers_stub():
    return 0

d2l.get_dataloader_workers = get_dataloader_workers_stub
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
print(f'# sentences: {len(sentences)}')
# sentences: 42069

vocab = d2l.Vocab(sentences, min_freq=10)
print(f'vocab size: {len(vocab)}')

#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    #[文本行][word]
    counter = collections.Counter([
        token for line in sentences for token in line])#摊平数据
    num_tokens = sum(counter.values())#总单词数


    # Return True if `token` is kept during subsampling
    def keep(token):#保留概率
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled)#直方图对长度计数，不是单词的频率


def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

print(compare_counts('the'))#高频词的量在下采样后大大下降
# of "the": before=50770, after=2068

corpus = [vocab[line] for line in subsampled]
corpus[:3]
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram.中间取上下文窗口的长度为随机数"""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line#一次加入一行数据，每个都是一个成员，注意区分append函数
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)#随机取样[1,max_window_size]
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
'''tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):#两个数组配对
    print('center', center, 'has contexts', context)'''#演示例子，暂时不用
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'

#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):#按概率抽取k个索引，从population里
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

generator = RandomGenerator([2, 3, 4])
print([generator.draw() for _ in range(10)])

#@save
def get_negatives(all_contexts, vocab, counter, K):#K是负样本数量，公式里有
    """Return noise words in negative sampling.抽取noise words"""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    #counter获取token对应的出现次数.0的数据是unk
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            #一个上下文词对应k个noise word
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives
all_negatives = get_negatives(all_contexts, vocab, counter, 5)

#all_contexts是上下文token列表的列表

#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]#填0，padding
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
        contexts_negatives), torch.tensor(masks), torch.tensor(labels))

x_1 = (1, [2, 2], [3, 3, 3, 3])#中心词,上下文词数组，Noise word数组
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)

#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)#subsampled是去掉unk之后的二维数组并进行下采样，去掉部分高频词，对应是行为line,列是每一个单词
    #counter是dict的子类，里面有每个token对应的出现次数
    corpus = [vocab[line] for line in subsampled]
    #vocab[line]是调用vocab的getitem方法，获取line所有元素的下标
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        '''可以在DataLoader内部生成返回的batch，必须实现 getitem len这两个函数'''
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab

if __name__ == '__main__':
    with d2l.Benchmark("k=110"):
        data_iter, vocab = load_data_ptb(512, 5, 5)
        for batch in data_iter:
            for name, data in zip(names, batch):
                print(name, 'shape:', data.shape)
            break

    plt.show()