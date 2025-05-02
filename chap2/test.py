import torch

def change(c):
    print(id(c))
    c[0]=1

a=[12,3]
print(id(a))
change(a)




