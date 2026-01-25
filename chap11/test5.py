import torch

valid_len = torch.tensor([3, 4])
maxlen=5
mask = (torch.arange((maxlen), dtype=torch.float32)[None, :])
valid_len=valid_len[:,None]
mask =  mask< valid_len
print(mask)