import math
import numpy as np
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


def f(x):
    return math.log(math.exp(x)+math.exp(x*-2-3))

def f_diff(x):
    return (math.exp(x)-2*math.exp(-2*x-3))/(math.exp(x)+math.exp(x*-2-3))

def compute(f,left,right):
    count=0
    mid = (right - left) / 2
    while right-left>1e-8:
        mid=(right+left)/2
        f_left = f(left)
        f_right = f(right)
        f_mid_diff=f_diff(mid)
        if f_mid_diff<0:
            left=mid
        else:
            right=mid
        count+=1
    return (f(mid),count)

(y,count)= compute(f,-5,5)
print(f"y={y},count={count}")

