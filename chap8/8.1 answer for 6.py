#8.1 answer 6
import time
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()

        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)


@d2l.add_to_class(AlexNet)
def fit_epoch(self):
    image_num=0
    start_time=time.time()
    self.gpu_memory_stats = []  # 格式: [(时间戳, batch_idx, gpu_id, allocated_mb, reserved_mb), ...]
    """Defined in :numref:`sec_linear_scratch`"""
    self.model.train()#设置为训练模式
    for batch_idx,batch in enumerate(self.train_dataloader):
        self._record_gpu_memory(batch_idx, stage="before_batch")
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
        X,y=self.prepare_batch(batch)
        image_num+=X.shape[0]
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
    end_time=time.time()
    speed_img=self.image_num/end_time-start_time


@d2l.add_to_class(AlexNet)
def _record_gpu_memory(self, batch_idx, stage):
    """辅助函数：记录当前所有GPU的内存使用情况"""
    timestamp = time.time()  # 记录时间戳
    num_gpus = torch.cuda.device_count()  # 获取GPU数量（你的代码中是2）

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):  # 切换到指定GPU
            # 已分配的内存（当前正在使用的）
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # 转换为MB
            # 已预留的内存（PyTorch为该进程缓存的总内存）
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # 转换为MB

            # 存储记录：时间戳、batch索引、GPU编号、分配的内存(MB)、预留的内存(MB)、阶段
            self.gpu_memory_stats.append({
                "timestamp": timestamp,
                "batch_idx": batch_idx,
                "gpu_id": gpu_id,
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "stage": stage  # 标记是训练前/后还是验证阶段
            })

def main():
    model = AlexNet(lr=0.01)
    data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
    trainer = d2l.Trainer(max_epochs=10,num_gpus=2)
    trainer.fit(model, data)
    df = pd.DataFrame(model.gpu_memory_stats)
    # 按GPU和阶段分组绘图
    for gpu_id in df['gpu_id'].unique():
        gpu_data = df[df['gpu_id'] == gpu_id]
        plt.figure(figsize=(10, 4))
        plt.plot(gpu_data['batch_idx'], gpu_data['allocated_mb'], label='Allocated (MB)')
        plt.plot(gpu_data['batch_idx'], gpu_data['reserved_mb'], label='Reserved (MB)')
        plt.title(f'GPU {gpu_id} Memory Usage')
        plt.xlabel('Batch Index')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.show()


if __name__ == "__main__":#确保代码块只在直接运行时执行，而在被导入时不执行
    main()