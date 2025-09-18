import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 生成大量标准正态分布的随机数
num_samples = 10000000
x_samples = np.random.randn(num_samples)

# 计算对应的累积分布函数值，即进行概率积分变换
y_samples = norm.cdf(x_samples)

# 绘制直方图，观察y_samples的分布情况
plt.hist(y_samples, bins=50, density=True, alpha=0.6, color='g')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Transformed Variable')
plt.show()