import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))

data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)

@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes!='object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()

print('5.7.5. Error Measure')

@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: torch.tensor(x.values.astype(float),
                                      dtype=torch.float32)#这个转换如果遇到one hot中的true,false，会相应的转换为1，0
    # Logarithm of prices
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               torch.log(get_tensor(data[label])).reshape((-1, 1)))  # Y
    #上面代码生成(X,Y)元组
    return self.get_tensorloader(tensors, train)

data = KaggleHouse(batch_size=64)
data.preprocess()
print(data.train.shape)
print('5.7.6.K-Fold Cross-Validation')

class MLP(d2l.Module):
    def __init__(self, input_dim, hidden_dim1,hidden_dim2,hidden_dim3, output_dim,lr,dropout_1,dropout_2):
        super(MLP, self).__init__()
        self.lr=lr
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_dim3, output_dim)  # 回归任务：无激活函数
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_pred, y_true):
        return nn.MSELoss()(y_pred, y_true)

def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k #//地板除法
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),
                                data.train.loc[idx]))
    return rets

def k_fold(trainer, data, k, lr):
    hidden_1=256
    hidden_2=256
    hidden_3=64
    lr=0.01
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = MLP(330,hidden_1,hidden_2,hidden_3,1,lr,0.2,0.3)
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))  # 获取验证损失
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models

trainer = d2l.Trainer(max_epochs=100)
models = k_fold(trainer, data, k=10, lr=0.01)
plt.show()


#用k个模型做预测,最后取均值
preds = [model(torch.tensor(data.val.values.astype(float), dtype=torch.float32))  for model in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = torch.exp(torch.cat(preds, 1)).mean(1)#torch.cat是连接函数，沿着维度1(列)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':ensemble_preds.detach().numpy()})
submission.to_csv('submission.csv', index=False)
