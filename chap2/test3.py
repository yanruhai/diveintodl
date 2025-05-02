import pandas as pd
import numpy as np

# 创建包含分类变量和缺失值的 DataFrame
data = pd.DataFrame({
    'color': ['red', 'blue', np.nan, 'green', 'red']
})

# 不考虑缺失值，dummy_na=False（默认情况）
dummies_false = pd.get_dummies(data['color'], dummy_na=False)
print("dummy_na=False 的结果：")
print(dummies_false)

# 考虑缺失值，dummy_na=True
dummies_true = pd.get_dummies(data['color'], dummy_na=True)
print("\ndummy_na=True 的结果：")
print(dummies_true)

print(np.arange(-1, -6, 1))