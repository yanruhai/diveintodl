import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def f(x, y):
    return (x**2 * y) / 5

def fx(x, y):
    return (2 * x * y) / 5

x = np.linspace(-3, 3, 30)
y = np.linspace(0, 3, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
slope_x = fx(X, Y)

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(221, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7, edgecolor='none')

y1, y2 = 1, 2
mask1 = np.abs(Y - y1) < 0.05
mask2 = np.abs(Y - y2) < 0.05

ax1.plot(X[mask1], Y[mask1], Z[mask1], 'r-', linewidth=3, label=f'y={y1}')
ax1.plot(X[mask2], Y[mask2], Z[mask2], 'b-', linewidth=3, label=f'y={y2}')

x0 = 2
for y0, color in [(1, 'red'), (2, 'blue')]:
    z0 = f(x0, y0)
    slope = fx(x0, y0)
    x_tan = np.array([x0-1, x0+1])
    z_tan = z0 + slope * (x_tan - x0)
    ax1.plot(x_tan, [y0, y0], z_tan, color=color, linewidth=2, linestyle='--')
    ax1.scatter([x0], [y0], [z0], color=color, s=50, edgecolors='black')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Surface f(x,y)=x^2*y/5, tangents at y=1 and y=2')
ax1.legend()
ax1.view_init(elev=25, azim=-60)

ax2 = fig.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(X, Y, slope_x, cmap=cm.coolwarm, alpha=0.7)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('fx = df/dx')
ax2.set_title('Eastward slope function fx(x,y)=2xy/5')
ax2.view_init(elev=25, azim=-60)

ax3 = fig.add_subplot(223)
contour = ax3.contour(X, Y, Z, levels=15, cmap=cm.viridis)
ax3.clabel(contour, inline=True, fontsize=8)
step = 3
Q = ax3.quiver(X[::step, ::step], Y[::step, ::step],
               fx(X, Y)[::step, ::step],
               np.zeros_like(X[::step, ::step]),
               angles='xy', scale_units='xy', scale=0.5, color='red', width=0.005)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Contours + eastward slope vectors (red arrows)')
ax3.axhline(y=1, color='red', linestyle='--', label='y=1')
ax3.axhline(y=2, color='blue', linestyle='--', label='y=2')
ax3.legend()

ax4 = fig.add_subplot(224)
y_vals = np.linspace(0, 3, 100)
x_fixed = 2
fx_vals = fx(x_fixed, y_vals)
ax4.plot(y_vals, fx_vals, 'g-', linewidth=2)
ax4.fill_between(y_vals, fx_vals, alpha=0.3)
ax4.set_xlabel('y')
ax4.set_ylabel(f'df/dx at x={x_fixed}')
ax4.set_title(f'At x={x_fixed}, eastward slope vs y\nSlope = f_xy = {2*x_fixed/5:.1f}')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

print("\n=== Numerical verification of f_xy ===")
print(f"f(x,y) = x^2 * y / 5")
print(f"df/dx = 2xy/5")
print(f"f_xy = d/dy (2xy/5) = 2x/5\n")

x0, y1, y2 = 2, 1, 2
print(f"At x={x0}:")
print(f"  y={y1}: eastward slope = {fx(x0, y1):.2f}")
print(f"  y={y2}: eastward slope = {fx(x0, y2):.2f}")
print(f"  Change = {fx(x0, y2) - fx(x0, y1):.2f}")
print(f"  f_xy = {2*x0/5:.2f} (slope of green line)")