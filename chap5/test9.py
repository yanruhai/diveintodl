from torchviz import make_dot
import  torch

# 定义变量和函数
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
f = x**2 * y + y**2

# 生成计算图可视化
dot = make_dot(f, params={'x': x, 'y': y})
dot.render("computation_graph", format="png")  # 保存为 PNG 文件