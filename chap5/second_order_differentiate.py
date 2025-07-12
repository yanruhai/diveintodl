import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)

# Define a simple MLP with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=2, output_dim=1):
        super(MLP, self).__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.randn(1, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.b2 = nn.Parameter(torch.randn(1, output_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1  # Hidden layer pre-activation
        H = self.sigmoid(Z1)         # Hidden layer activation
        O = H @ self.W2 + self.b2    # Output layer
        return O

# Create synthetic data
n, d, q = 3, 2, 1  # batch size, input dim, output dim
X = torch.randn(n, d, requires_grad=True)  # Input: 3 samples, 2 features
Y = torch.randn(n, q)                      # Target: 3 samples, 1 output

# Initialize model
model = MLP(input_dim=d, hidden_dim=2, output_dim=q)

# Forward pass
output = model(X)
loss = torch.mean(0.5 * (output - Y) ** 2)  # Mean squared error loss

# First-order gradient w.r.t. W1[0,0]
grads = torch.autograd.grad(loss, model.W1, create_graph=True)[0]#返回值是多个对变量的梯度
'''torch.autograd.grad(
    outputs,           # 需要求导的输出张量（通常是损失函数）
    inputs,            # 需要计算梯度的输入张量
    grad_outputs=None, # 输出张量的梯度（用于链式法则）
    retain_graph=None, # 是否保留计算图
    create_graph=False,# 是否创建用于高阶导数的计算图
    only_inputs=True,  # 是否只返回输入的梯度
    allow_unused=False # 是否允许未使用的输入
)'''
grad_W1_00 = grads[0, 0]  # Gradient of loss w.r.t. W1[0,0]

# Second-order gradient w.r.t. W1[0,0]
second_grad = torch.autograd.grad(grad_W1_00, model.W1, retain_graph=True)[0]
second_grad_W1_00 = second_grad[0, 0]  # Second derivative d^2 loss / d W1[0,0]^2

# Print results
print(f"Loss: {loss.item():.4f}")
print(f"First derivative dL/dW1[0,0]: {grad_W1_00.item():.4f}")
print(f"Second derivative d^2L/dW1[0,0]^2: {second_grad_W1_00.item():.4f}")