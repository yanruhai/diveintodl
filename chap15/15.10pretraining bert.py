import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)

net = d2l.BERTModel(len(vocab), num_hiddens=768,
                    ffn_num_hiddens=1024, num_heads=12, num_blks=12, dropout=0.2)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()

#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

'''def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net(*next(iter(train_iter))[:4])
    #net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    net=net.to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')'''


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    device = devices[0]
    net = net.to(device)

    trainer = torch.optim.Adam(net.parameters(), lr=0.01)

    # ⚡ 修正警告：改为新版标准的写法
    scaler = torch.amp.GradScaler('cuda')

    step, timer = 0, d2l.Timer()
    metric = d2l.Accumulator(4)

    print(f"🚀 5080 引擎已启动，正在执行 {num_steps} 个 Step 的闪电训练...")

    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, \
                mlm_weights_X, mlm_Y, nsp_y in train_iter:

            # 搬运数据到 GPU
            tokens_X = tokens_X.to(device)
            segments_X = segments_X.to(device)
            valid_lens_x = valid_lens_x.to(device)
            pred_positions_X = pred_positions_X.to(device)
            mlm_weights_X = mlm_weights_X.to(device)
            mlm_Y, nsp_y = mlm_Y.to(device), nsp_y.to(device)

            trainer.zero_grad()
            timer.start()

            # ⚡ 修正警告：改为新版标准的 torch.amp.autocast('cuda')
            with torch.amp.autocast('cuda'):
                mlm_l, nsp_l, l = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                    pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)

            # 混合精度反向传播
            scaler.scale(l).backward()
            scaler.step(trainer)
            scaler.update()

            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()

            step += 1
            if step % 10 == 0:
                print(f"进度: [{step}/{num_steps}]...")

            if step == num_steps:
                num_steps_reached = True
                break

    print("\n" + "=" * 30)
    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on {str(device)}')
    print("=" * 30)

# ⚡ 核心解药：修正函数名，将 d2l._sequence_mask 替换为标准的 d2l.sequence_mask
def safe_masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 🔔 注意这里：去掉了前面的下划线，改成了 d2l.sequence_mask
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-30000.0)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# ⚡ 重新绑定补丁
d2l.masked_softmax = safe_masked_softmax

if __name__ == '__main__':
    train_bert(train_iter, net, loss, len(vocab), devices, 50)
    plt.show()