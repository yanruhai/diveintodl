import  torch

i=2
x1 = torch.tensor(0, dtype=torch.float32)
while True:
   t=x1
   x1=torch.exp(x1)
   print(x1)
   if x1==0:
       break
   x1=t-0.5



