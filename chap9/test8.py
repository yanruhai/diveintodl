import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

@d2l.add_to_class(d2l.TimeMachine)
def _download(self):
    fname = d2l.download('https://www.gutenberg.org/ebooks/1342.txt.utf-8', self.root,
                         '090b5e7e70c295757f55df93cb0a180b9691891a')
    with open(fname,encoding='utf-8') as f:
        return f.read()
predict_data=d2l.TimeMachine(batch_size=1024, num_steps=32)