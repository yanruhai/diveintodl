import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l

class BiRNNScratch(d2l.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.lr=1
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn_f = d2l.RNNScratch(embed_size, num_hiddens, sigma)
        self.rnn_b = d2l.RNNScratch(embed_size, num_hiddens, sigma)
        self.linear = nn.Linear(num_hiddens * 2, vocab_size)

    def forward(self, X, state=None):
        embs = self.embedding(X.T).permute(1, 0, 2)  # (T, B, E)
        f_state, b_state = (None, None) if state is None else state

        f_outputs, f_state = self.rnn_f(embs, f_state)                   # list[T]
        b_outputs, b_state = self.rnn_b(embs.flip(0), b_state)           # list[T] (已反向)
        b_outputs = b_outputs[::-1]                                      # 恢复正序

        outputs = [torch.cat((f, b), dim=-1) for f, b in zip(f_outputs, b_outputs)]
        outputs = torch.stack(outputs)                   # (T, B, 2*H)
        outputs = self.linear(outputs)                   # (T, B, V)
        outputs = outputs.permute(1, 0, 2).contiguous()  # (B, T, V)
        return outputs, (f_state, b_state)

    def loss(self, Y_hat, Y):
        # Y_hat: (B, T, V), Y: (B, T)
        # 关键：两者都切到长度 T-1，且用 contiguous().view 彻底避免 view 共享内存问题
        return F.cross_entropy(
            Y_hat[:, :-1, :].contiguous().view(-1, Y_hat.shape[-1]),
            Y[:, 1:].contiguous().view(-1),
            reduction='mean'
        )

    def training_step(self, batch):
        X, Y = batch
        Y_hat, _ = self(X)
        l = self.loss(Y_hat, Y)
        self.plot('ppl', torch.exp(l.detach()), train=True)
        return l

    def validation_step(self, batch):
        X, Y = batch
        Y_hat, _ = self(X)
        l = self.loss(Y_hat, Y)
        self.plot('ppl', torch.exp(l.detach()), train=False)
        return l


# ====================== 直接复制运行 ======================
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
model = BiRNNScratch(len(data.vocab), embed_size=32, num_hiddens=128)

trainer = d2l.Trainer(max_epochs=200, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)