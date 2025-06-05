import numpy as np

a_x=range(1,100)
for a in a_x:
    r= np.log(np.exp(a)+1)-a
    print(r)
