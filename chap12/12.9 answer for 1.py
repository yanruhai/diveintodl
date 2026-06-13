import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.3}, data_iter)
plt.show()