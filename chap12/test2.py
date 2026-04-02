

results = [(-5, -2), (-4, -1.6), (-3.2, -1.28)]

# 错误写法：无*，zip只收到1个参数（整个results列表）
wrong_zip = zip(results)
print(*(zip(*results)))  # 输出：[((-5, -2),), ((-4, -1.6),), ((-3.2, -1.28),)]