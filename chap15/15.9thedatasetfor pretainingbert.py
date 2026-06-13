import os
import random
import torch
from d2l import torch as d2l

#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r',encoding='utf-8') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones,a spece is inserted before period,
    #so this uses space.space to split
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]#lines内每一行是一个段落,用于训练时表示上下文
    random.shuffle(paragraphs)#只打乱段落之间，段落内部不打乱
    return paragraphs

#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    '''paragraphs是语料库，paragraph是单个的段落,段落内有很多sentence，是个列表'''
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:#for aligning to training task afterward
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)#d2l的库用来生成<cls>token1<sep>token2<sep>结构,上一节有源码
        #segments里面结构是[0,0,0,1,1,1,1]标识是第一段话还是第二段话，用于掩码
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
#库内代码转出
'''def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其段标号"""
    # 1. 强行在最前端焊上 <cls>，并把第一句话放进来
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    
    # 2. 第一句话对应的段编码全都是 0
    # 比如 tokens_a 长度是 5，那 segments 就是 [0, 0, 0, 0, 0, 0, 0]（算上 cls 和 sep）
    segments = [0] * len(tokens)
    
    # 3. 如果有第二句话（NSP 任务里肯定有 tokens_b）
    if tokens_b is not None:
        # 物理拼接：追加第二句话的内容，并在最后封顶一个 <sep>
        tokens += tokens_b + ['<sep>']
        # 段编码追加：第二句话以及它末尾的 sep，对应的编码全都是 1
        segments += [1] * (len(tokens_b) + 1)
        
    return tokens, segments'''

#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # For the input of a masked language model, make a new copy of tokens and
    # replace some of them by '<mask>' or random tokens
    #tokens参数是原始句子单词数组
    mlm_input_tokens = [token for token in tokens]#复制一个新的tokens数组，用于后序处理，原数组保留
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)#打乱位置列表
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:#80%需要预测的
            masked_token = '<mask>'#替换成<mask>标签
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:#10%原始数据不变
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)#idx_to_token是列表，下标是id,值是token
                #random.sample(列表，数量)多选，这里的choice只能单选
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels#mlm_input_tokens返回已经换过<mask>标记的tokens数组
    #pred_positions_and_labels保存需要预测的位置和标记，里面是元组数组

#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []#不包括特殊标记的token列表
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)#排除特殊标记<cls><sep>后的下标列表
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))#15%需要预测
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

#@save
def _pad_bert_inputs(examples, max_len, vocab):
    #examples为五元组列表 前3元是用于mlm预测，(token_ids,pred_positions,mlm_pred_label_ids)
    #用来对齐token列表,并且将补充的Padding做备注。得到7元组。
    #_get_nsp_data_from_paragraph这个函数里会处理句子长度>max_len的情况
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        #mlm_weights内元素1表示真实需要预测的掩码位置，0表示后面补的padding位置，计算时忽略，不回传梯度
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))#
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]#将文本拆成token数组,paragraphs是一整段的列表
        #word是按单词拆，如果token=‘char’，按字母拆。tokenize函数会多出一层列表，所以结果是三层列表的word级的token
        # nsp_data_from_paragraph.append((tokens, segments, is_next))引用代码。
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]#3层嵌套列表拆出来一层
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])#vocab接受由token组成的列表的列表
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))#extend函数是list的函数，可以平铺返回的列表数据
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]#生成五元组
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = "../data/wikitext-2"
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    #train_iter = torch.utils.data.DataLoader(train_set, batch_size,shuffle=True, num_workers=num_workers)
    train_iter = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # ⚡ 强制改成 0！变成纯单进程单线程模型
    )
    return train_iter, train_set.vocab

if __name__ == '__main__':
    batch_size, max_len = 128, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
         mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
              pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
              nsp_y.shape)
        break
    len(vocab)