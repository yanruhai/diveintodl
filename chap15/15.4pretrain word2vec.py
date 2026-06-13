import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
import d2l.torch as d2l

# 1. 强制重写库函数的返回值，让它在你的脚本运行期间始终返回 0
def get_dataloader_workers_stub():
    return 0

d2l.get_dataloader_workers = get_dataloader_workers_stub
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(embed(x))#对embed的数据做索引查找

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    '''用embed_v,就是nn.embedding类来处理center数据，embed_u处理context_negative数据'''
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

print(skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape)

class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)#按照第一维求平均值

loss = SigmoidBCELoss()
pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)#(1,4)*2后变成(2,4)
#pred有两行训练数据，行中每个数据是中心词和上下文词或noise的点积结果，因为labe只有一个1，说明上下文词
#只有1个，后面3个是noise .1.1 Upos^TVcenter,2.2 Uneg1^TVcenter...
label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
#将这个数据代入到binary_cross_entropy_with_logits里面计算损失，loss=-[ylog(δ(x))+(1-y)(1-δ(x))]
#里面的y是取自当前pred数据对应的label数据，用来表示是否是中心词与上下文的点积结果
print(loss(pred,label,mask)* mask.shape[1])
print(loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))

def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')

embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))

def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            #获取pred向量，就是用中心词与上下文词或noise词乘积
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            #l.sum()：把当前这个 Batch 里所有样本的损失加起来，存入 metric[0]。
            #l.numel()：统计当前这个 Batch 贡献了多少个训练样本，存入 metric[1]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')

if __name__ == '__main__':
    lr, num_epochs = 0.002, 5
    train(net, data_iter, lr, num_epochs)

    def get_similar_tokens(query_token, k, embed):
        W = embed.weight.data#权重向量,(6719,100)
        x = W[vocab[query_token]]#vocab[query_token]值是1045，表示取出W的第1045行的数据
        ch=vocab[query_token]
        # Compute the cosine similarity. Add 1e-9 for numerical stability
        cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                          torch.sum(x * x) + 1e-9)#这里W,x的向量积算出当前token与所有token的相似度
        topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
        #torch.topk(input, k) 的返回结果包含两类数据：索引 [0] (Values)：前k个最大的数值（在这个场景下是最
        # 高的余弦相似度分数）。索引 [1] (Indices)：前 k 个最大值对应的索引（ID）。
        for i in topk[1:]:  # Remove the input words
            print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')


    get_similar_tokens('chip', 4, net[0])
    plt.show()