import d2l
import torch
from d2l.torch import SyntheticRegressionData


#Implement a data generator that produces new data on the fly, every time the iterator is called.

@d2l.add_to_class(SyntheticRegressionData)
def create_new_data(self,batch):#生成batch条数据加入原数据后面
    Xtemp = torch.randn(batch, len(self.w))
    torch.cat((self.X, Xtemp), dim=0)
    noiseTemp = torch.randn(batch, 1) * self.noise
    torch.cat((self.noise, noiseTemp), dim=0)
    yTemp = torch.matmul(self.X, self.w.reshape((-1, 1))) + self.b + self.noise
    torch.cat(self.y, yTemp)

