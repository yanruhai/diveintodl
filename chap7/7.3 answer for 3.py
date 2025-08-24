import torch
from torch import nn

def normalize_padding(padding):
    if isinstance(padding, int):
        padding = (padding, padding)
        # 情况2：如果是列表或元组，取前两个元素（不足则补全）
    elif isinstance(padding, (list, tuple)):
        # 只取前两个元素，多余的忽略
        pad_list = list(padding)[:2]
        # 如果长度不足2，用最后一个元素补全（如(3,) → (3,3)）
        while len(pad_list) < 2:
            pad_list.append(pad_list[-1] if pad_list else 0)
        padding = tuple(pad_list)
    return padding

def mirror_extend_padding(X,padding=(1,1)):#padding为一个二元元组，第一个元素表示高度上单个方向上的padding值
    padding=normalize_padding(padding)
    print(X)
    h_up=X[0,:].reshape(1,X.shape[1])
    h_down=X[-1,:].reshape(1,X.shape[1])
    h_up=h_up.reshape((1,X.shape[1]))
    for _ in range(padding[0]):
        X=torch.cat([h_up,X],dim=0)
        X=torch.cat([X,h_down],dim=0)

    w_left=X[:,0].reshape(X.shape[0],1)
    w_right=X[:,-1].reshape(X.shape[0],1)
    for _ in range(padding[1]):
        X = torch.cat([w_left, X], dim=1)
        X = torch.cat([X,w_right], dim=1)
    print(X)


X = torch.rand(size=(4, 4))
padding=1
mirror_extend_padding(X,padding=(2,2))
