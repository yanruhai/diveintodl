import collections
import random
import re
import torch

text = "hello world"
my_list=list(text)
counter = collections.Counter(my_list)
token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
