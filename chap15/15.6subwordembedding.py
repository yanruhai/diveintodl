import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']

raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():#类似与map的迭代
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]#加空格是为了后面的split成单个字母，方便迭代
print(token_freqs)

def get_max_freq_pair(token_freqs):
    '''在已知token_freqs库中遍历所有词，迭代生成临时词频类型dict的pairs,存储类似{"a b":4,...}'''
    pairs = collections.defaultdict(int)#自动存储默认值,普通字典取没有的值会报错，这种会存进去
    for token, freq in token_freqs.items():
        symbols = token.split()#根据空格拆字符
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value

def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))#把列表转成字符串
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))#如果token中没有maxfreqpair数据，返回原值
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs

num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
print(symbols)

def segment_BPE(tokens, symbols):
    '''根据前面几个函数构造出的symbols字典，挨个对token元素做分析，是否存在词素'''
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:#如果在词典里找到对应的token
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs

tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))