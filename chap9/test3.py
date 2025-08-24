import torch



T=1000
tau=4
time = torch.arange(1, 1000 + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.randn(T) * 0.2
features = [x[i : T-tau+i] for i in range(tau)]#生成T*4的张量
# 通过索引修改列表中的原始元素
'''for i in range(len(features)):
    # 直接对列表的第i个元素重新赋值（reshape后的新张量）
    features[i] = features[i].reshape(1, -1)  # 改为 (1, 996) 形状'''
features = torch.stack(features, 0)#该张量在列上堆
print()