import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    '''X.shape=(batch_size, num_queries, num_kv)'''
    # X: 3D tensor, valid_lens: 1D or 2D tensor,if 2d ,valid_lens∈(batch_size, num_queries)
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        #[None,:]表示在0维插入一个新维度，大小为1，等价于torch.unsqueeze(0),用于以后广播.test2是测试代码，可以参考
        X[~mask] = value#对于X中掩码位置根据bool赋值
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)#-1表示负索引约束，表示张量的最后一个维度
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            '''>>> x = torch.tensor([1, 2, 3])
        >>> x.repeat_interleave(2)
        tensor([1, 1, 2, 2, 3, 3])
        >>> y = torch.tensor([[1, 2], [3, 4]])
        >>> torch.repeat_interleave(y, 2),这里2是重复次数,会先展开，然后重复
        tensor([1, 1, 2, 2, 3, 3, 4, 4])'''
        else:
            valid_lens = valid_lens.reshape(-1)#-1是展平的意思
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        #X.reshape(-1, shape[-1]) shape[-1]是保持最后一维，其他维展开成一维,再和展开的valid_len做掩码运算
        return nn.functional.softmax(X.reshape(shape), dim=-1)#reshape(shape)就是还原张量维度
    #dim=-1表示以行为单位做softmax,比如[[1,2,3], [4,5,6]],softmax(dim=-1)后为:
    #[[0.0900, 0.2447, 0.6652],
        #[0.0900, 0.2447, 0.6652]]

masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))

Q = torch.ones((2, 3, 4))
K = torch.ones((2, 4, 6))
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))

class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)#bmm只对数据后两维做乘法, scores=(2,1,2)*(2,2,10)=(2,1,10)
        self.attention_weights = masked_softmax(scores, valid_lens)
        print("weights:",self.attention_weights)
        test=self.dropout(self.attention_weights)
        print("dropuout:",test)
        return torch.bmm(self.dropout(self.attention_weights), values)#train模式下dropout不生效


queries = torch.normal(0, 1, (2, 1, 2))
keys = torch.normal(0, 1, (2, 10, 2))
values = torch.normal(0, 1, (2, 10, 4))
valid_lens = torch.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval()#进入val验证模式
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))

'''d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')'''

class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        test=self.w_v(features)#(2,1,10,8)->(2,1,10,1)
        scores = self.w_v(features).squeeze(-1)#remove the last dimention,this must be 1
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
plt.show()