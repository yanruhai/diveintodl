import torch
import torch.nn as nn

# 1. Input data (3 elements in 1D, treat as a tiny feature map)
x = torch.tensor([[[1.0, 3.0, 2.0]]])  # Shape: [batch=1, channel=1, length=3]

# 2. Step 1: Compute max(x1, x2) with ReLU + “pseudo - convolution” (1x1 conv here just for structure)
#    Concept: Use 1x1 conv to keep elements, then ReLU for pairwise max
conv1x1 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
# Manually set weights to [1, -1, 0] to compute x1 - x2 (first two elements)
conv1x1.weight = torch.nn.Parameter(torch.tensor([[[1.0, - 1.0, 0.0]]]))
out1 = conv1x1(x)  # out1 = x1 - x2
relu1 = nn.ReLU()
out1_relu = relu1(out1)  # ReLU(x1 - x2)
# Add back x2 to get max(x1, x2): max(x1,x2) = ReLU(x1 - x2) + x2
# To add x2, we can use another 1x1 conv with weights [0, 1, 0] to extract x2
conv_add_x2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
conv_add_x2.weight = torch.nn.Parameter(torch.tensor([[[0.0, 1.0, 0.0]]]))
x2 = conv_add_x2(x)
max_x1x2 = out1_relu + x2  # Now max_x1x2 holds max(x1, x2)

# 3. Step 2: Compute max(max(x1,x2), x3) with ReLU + convolution
#    Compute max_x1x2 - x3
conv_sub_x3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
conv_sub_x3.weight = torch.nn.Parameter(torch.tensor([[[1.0, 0.0, - 1.0]]]))
out2 = conv_sub_x3(max_x1x2)  # out2 = max(x1,x2) - x3
relu2 = nn.ReLU()
out2_relu = relu2(out2)  # ReLU(max(x1,x2) - x3)
# Add back x3 to get final max
conv_add_x3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
conv_add_x3.weight = torch.nn.Parameter(torch.tensor([[[0.0, 0.0, 1.0]]]))
x3 = conv_add_x3(x)
max_final = out2_relu + x3  # Now max_final = max(x1, x2, x3)

print("Input:", x)
print("Max of 3 elements:", max_final)