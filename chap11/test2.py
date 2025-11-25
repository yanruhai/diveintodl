import torch

maxlen=10
valid_len=torch.tensor([3, 5, 2])
print(valid_len[:,None])
print(torch.arange((maxlen), dtype=torch.float32)[None, :])
mask = torch.arange((maxlen), dtype=torch.float32)[None, :] < valid_len[:, None]
#print(mask)