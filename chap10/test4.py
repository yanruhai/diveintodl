example = "hello  world <eos>"  # 注意hello和world之间是两个空格
tokens = example.split(' ')
print(tokens)  # 输出：['hello', '', 'world', '<eos>']