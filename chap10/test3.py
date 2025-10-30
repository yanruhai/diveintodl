import multiprocessing
import time
import random  # 模拟训练分数


def train_model(params):
    """单个超参数组合的训练函数（网格搜索任务）"""
    lr, batch_size = params
    print(f"Training with lr={lr}, batch_size={batch_size}")
    time.sleep(1)  # 模拟 1 秒训练时间
    score = random.uniform(0.7, 0.95)  # 模拟分数
    print(f"Done with lr={lr}, batch_size={batch_size}, score={score:.3f}")
    return (params, score)


def cleanup(best_params, all_scores):
    """所有进程结束后执行的函数"""
    print(f"Grid search complete! Best params: {best_params}, Best score: {max(all_scores):.3f}")
    # 这里可以加保存模型、绘图等代码


if __name__ == '__main__':
    # 定义网格参数
    lrs = [0.01, 0.1, 0.001]
    batch_sizes = [16, 32, 64]
    grid_params = [(lr, bs) for lr in lrs for bs in batch_sizes]  # 展开成9个任务

    # 创建进程池（进程数设为CPU核心数的一半，避免过度并行）
    num_processes = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=num_processes)

    # 并行执行网格搜索
    results = pool.map(train_model, grid_params)

    # 等待所有进程结束
    pool.close()
    pool.join()

    # 提取结果，找到最佳
    all_scores = [score for _, score in results]
    best_idx = all_scores.index(max(all_scores))
    best_params = results[best_idx][0]

    # 结束后调用清理函数
    cleanup(best_params, all_scores)