import  torch

dec_valid_lens = torch.arange(1, 9 + 1).repeat(2,1)
print(dec_valid_lens)