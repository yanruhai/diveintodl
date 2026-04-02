import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 网格
x = np.linspace(-1.8, 1.8, 60)
y = np.linspace(-1.8, 1.8, 60)
X, Y = np.meshgrid(x, y)

# 曲面：z = x + 0.6 y² - 0.35  (中间穿过 z=0)
Z = X + 0.6 * Y**2 - 0.35

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 画曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, linewidth=0)

# 画 z=0 平面（半透明灰色）
ax.plot_surface(X, Y, np.zeros_like(X), color='lightgray', alpha=0.35)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('z = x + 0.6 y² - 0.35\n(类似图11.1的局部弯曲穿过 z=0)')

# 视角调整（让交线看起来像图中那样弯曲）
ax.view_init(elev=22, azim=135)

# 添加颜色条（可选）
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

Z = X + 0.7 * Y**2 - 0.4 + 0.25 * np.sin(4 * X + Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.82, linewidth=0)
ax.plot_surface(X, Y, np.zeros_like(X), color='lightgray', alpha=0.3)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('z = x + 0.7 y² - 0.4 + 0.25 sin(4x + y)')

ax.view_init(elev=20, azim=140)
fig.colorbar(surf, ax=ax, shrink=0.6)

plt.show()