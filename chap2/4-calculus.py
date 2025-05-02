import numpy as np


def f(x):
    return 3 * x ** 2 - 4 * x

for h in 10.0**np.arange(-1, -15, -1):
    print(f'h={h:.30f}, numerical limit={(f(1+h)-f(1))/h:.5f}')

print('2.4.2. Visualization Utilities')
