import torch

# ====================== 1. 模拟加性注意力的原始张量 ======================
# 你的场景：batch=128, query_len=1, key_len=9, num_hiddens=256
batch_size = 128
query_len = 1    # 解码器单步query长度
key_len = 9      # 编码器key长度（时间步）
num_hiddens = 256# 隐藏层维度

# 模拟经过W_q/W_k变换后的queries和keys（维度不变）
queries = torch.randn(batch_size, query_len, num_hiddens)  # (128, 1, 256)
keys = torch.randn(batch_size, key_len, num_hiddens)       # (128, 9, 256)

print("原始维度：")
print(f"queries: {queries.shape} → (batch, query_len, num_hiddens)")
print(f"keys: {keys.shape} → (batch, key_len, num_hiddens)")
print("-" * 50)

# ====================== 2. 测试：直接相加（报错/结果不符合预期） ======================
print("测试1：直接相加 → 要么报错，要么维度丢失")
try:
    direct_add = queries + keys
    print(f"直接相加结果维度：{direct_add.shape} → 丢失query_len维度，不符合需求！")
except RuntimeError as e:
    print(f"直接相加报错：{e}")
print("-" * 50)

# ====================== 3. 测试：扩维后相加（正确方式） ======================
print("测试2：扩维后相加 → 维度正确，满足加性注意力需求")
# 给queries新增第2维（索引从0开始），keys新增第1维
queries_expand = queries.unsqueeze(2)  # (128, 1, 1, 256)
keys_expand = keys.unsqueeze(1)        # (128, 1, 9, 256)

print(f"queries扩维后：{queries_expand.shape} → (batch, query_len, 1, num_hiddens)")
print(f"keys扩维后：{keys_expand.shape} → (batch, 1, key_len, num_hiddens)")

# 扩维后相加（广播机制生效）
add_result = queries_expand + keys_expand
print(f"扩维相加结果维度：{add_result.shape} → (batch, query_len, key_len, num_hiddens)")
print("-" * 50)

# ====================== 4. 模拟加性注意力下一步：压缩为相似度分数 ======================
print("测试3：模拟加性注意力计算相似度分数")
# 经过tanh激活（模拟加性注意力的tanh(W_q q + W_k k)）
tanh_result = torch.tanh(add_result)  # (128, 1, 9, 256)

# 用线性层压缩最后一维（num_hiddens→1），得到相似度分数
W_v = torch.nn.Linear(num_hiddens, 1, bias=False)
score = W_v(tanh_result)  # (128, 1, 9, 1)
score = score.squeeze(-1) # 去掉最后一维 → (128, 1, 9)，这就是query对每个key的相似度

print(f"相似度分数维度：{score.shape} → (batch, query_len, key_len)")
print(f"单样本相似度分数示例：\n{score[0][0].detach().numpy()}")  # 打印第一个样本的1×9相似度分数