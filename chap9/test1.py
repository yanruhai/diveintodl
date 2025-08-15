import torch

T=1000
tau=4
time = torch.arange(1, 1000 + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.randn(T) * 0.2
features = [x[i : T-tau+i] for i in range(tau)]#生成T*4的张量
features = torch.stack(features, 1)#该张量在列上堆

multistep_preds = torch.zeros(T)
multistep_preds[:] = x
for i in range(500 + tau, T):
    print(multistep_preds[i - tau:i].reshape((1, -1)))
multistep_preds = multistep_preds.detach().numpy()

features2=[x[i:i+4] for i in range(T-tau)]
features2=torch.stack(features2,0)
print()