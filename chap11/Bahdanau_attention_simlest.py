import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Bahdanau 加性注意力（Bahdanau Attention / Additive Attention）
    核心：q(解码器) → k/v(编码器)，加性相似度计算 + softmax 权重 + 加权求和v
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        # 可学习参数：加性变换线性层 + 注意力得分映射层
        # W_a: 拼接(q, k)后做线性变换，输入维度=dec_hid_dim + enc_hid_dim
        self.W_a = nn.Linear(dec_hid_dim + enc_hid_dim, dec_hid_dim, bias=False)
        # v_a: 将变换后的特征映射为标量注意力得分
        self.v_a = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_hiddens):
        """
        前向传播：核心计算流程
        :param dec_hidden: 解码器当前隐藏状态 (q) → [batch_size, dec_hid_dim]
        :param enc_hiddens: 编码器所有隐藏状态 (k/v) → [batch_size, enc_seq_len, enc_hid_dim]
        :return: context_vec(上下文向量), attn_weights(注意力权重)
        """
        batch_size = enc_hiddens.shape[0]
        enc_seq_len = enc_hiddens.shape[1]

        # 步骤1：扩展q的维度，与k对齐（便于拼接）→ [batch_size, enc_seq_len, dec_hid_dim]
        # 让解码器单个时间步的q，能和编码器所有时间步的k逐一计算相似度
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, enc_seq_len, 1)

        # 步骤2：加性拼接变换，计算原始注意力得分（e_ti）
        # 拼接q和k → [batch_size, enc_seq_len, dec_hid_dim + enc_hid_dim]
        # 经过W_a + tanh → [batch_size, enc_seq_len, dec_hid_dim]
        # 经过v_a → [batch_size, enc_seq_len, 1]（每个k对应一个得分）
        energy = self.v_a(torch.tanh(self.W_a(torch.cat((dec_hidden, enc_hiddens), dim=2))))
        # 去除最后一维，便于softmax → [batch_size, enc_seq_len]
        energy = energy.squeeze(2)

        # 步骤3：softmax归一化，得到注意力权重（α_ti）→ 和为1
        attn_weights = F.softmax(energy, dim=1)  # [batch_size, enc_seq_len]

        # 步骤4：权重加权求和v，得到上下文向量（c_t）
        # 权重扩展维度 → [batch_size, 1, enc_seq_len]
        # 与enc_hiddens(k/v)做矩阵乘法 → [batch_size, 1, enc_hid_dim]
        # 去除中间维度 → [batch_size, enc_hid_dim]
        context_vec = torch.bmm(attn_weights.unsqueeze(1), enc_hiddens).squeeze(1)

        return context_vec, attn_weights

# ------------------- 测试代码：验证维度匹配和前向传播 -------------------
if __name__ == "__main__":
    # 超参数设置（贴合Seq2Seq实际场景）
    BATCH_SIZE = 4        # 批次大小
    ENC_SEQ_LEN = 10      # 编码器序列长度（输入序列长度）
    DEC_HID_DIM = 128     # 解码器隐藏层维度
    ENC_HID_DIM_SINGLE = 64# 编码器单向隐藏层维度
    ENC_HID_DIM = 2 * ENC_HID_DIM_SINGLE  # 双向编码器，最终隐藏维度=2*单向

    # 1. 初始化注意力层
    attn = BahdanauAttention(enc_hid_dim=ENC_HID_DIM, dec_hid_dim=DEC_HID_DIM)

    # 2. 构造输入：严格匹配q/k/v来源和形状
    dec_hidden = torch.randn(BATCH_SIZE, DEC_HID_DIM)  # q：解码器当前隐藏态
    enc_hiddens = torch.randn(BATCH_SIZE, ENC_SEQ_LEN, ENC_HID_DIM)  # k/v：编码器所有隐藏态

    # 3. 前向传播
    context_vec, attn_weights = attn(dec_hidden, enc_hiddens)

    # 4. 打印输出维度（验证正确性）
    print("上下文向量维度 (context_vec):", context_vec.shape)  # 预期：[4, 128]
    print("注意力权重维度 (attn_weights):", attn_weights.shape)  # 预期：[4, 10]
    print("注意力权重每行和 (验证softmax):", attn_weights.sum(dim=1))  # 预期：每行≈1.0