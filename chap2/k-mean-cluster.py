import pandas as pd
from scipy.stats import expon
import numpy as np

def get_data():
    data=[]
    a1 = np.random.rand()
    data.append(a1)
    a2= np.random.uniform(1, 5)
    data.append(a2)
    a3 = expon.rvs(scale=15)
    data.append(a3)
    a4= np.random.poisson(2)
    data.append(a4)
    a5=np.random.rand()
    data.append(a5)
    return data

def assigned(single_data,centroids ,error):#将当个数据点分配到centroids 某个数据中
    print()


df =  pd.DataFrame(columns=['a1','a2','a3','a4','a5'])
for i in range(1000):
    df.loc[i]=get_data()
random_indices = np.random.choice(df.index, 10, replace=False)
random_rows = df.loc[random_indices]
df.drop(random_rows)


print()
