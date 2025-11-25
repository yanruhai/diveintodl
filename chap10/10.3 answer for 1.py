import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i==0 else num_hiddens, num_hiddens, sigma)#num_inputs if i==0 else num_hiddens 这里是一个语句
                                    for i in range(num_layers)])#产生深度为num_layers的RNNScratch列表


@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs
    if Hs is None: Hs = [None] * self.num_layers
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])#这里的i表示深度
        outputs = torch.stack(outputs, 0)
    return outputs, Hs

@d2l.add_to_class(d2l.TimeMachine)
def _download2(self):
    fname = d2l.download('https://www.gutenberg.org/cache/epub/77146/pg77146.txt', self.root,
                         '090b5e7e70c295757f55df93cb0a180b9691891a')#Buddhism & science by Paul Dahlke
    with open(fname,encoding='utf-8') as f:
        return f.read()
    'https://www.gutenberg.org/cache/epub/1342/pg1342.txt'

@d2l.add_to_class(d2l.TimeMachine)
def _download3(self):
    fname = d2l.download('https://www.gutenberg.org/cache/epub/1342/pg1342.txt', self.root,
                         '090b5e7e70c295757f55df93cb0a180b9691891a')#Pride and Prejudice by Jane Austen
    with open(fname,encoding='utf-8') as f:
        return f.read()

@d2l.add_to_class(d2l.TimeMachine)
def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        """Defined in :numref:`sec_language-model`"""
        super(d2l.TimeMachine, self).__init__()
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download()+self._download2()+self._download3())
        print('corpus长度为',len(corpus))
        array = d2l.tensor([corpus[i:i+num_steps+1]
                            for i in range(len(corpus)-num_steps)])
        self.X, self.Y = array[:,:-1], array[:,1:]

data = d2l.TimeMachine(batch_size=1024, num_steps=32)
rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                              num_hiddens=32, num_layers=2)
model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
trainer = d2l.Trainer(max_epochs=200, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#plt.show()

#concise implementation

class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)

class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens,num_layers):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens,num_layers)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)

lstm=LSTM(num_inputs=len(data.vocab), num_hiddens=32,num_layers=2)
gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)
plt.show()