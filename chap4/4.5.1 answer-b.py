import  torch
# 创建FP32类型的单个变量（运算时自动转为TF32）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
x8 = torch.tensor(3.14159, dtype=torch.float32).cuda()
print(x8)