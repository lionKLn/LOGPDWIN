import numpy as np


# 不确定性采样
def uncertainty_sampling(probs, n_samples):
    uncertainty = np.abs(probs - 0.5)
    indices = np.argsort(uncertainty)[:n_samples]
    return indices


# 随机采样（对比用）
def random_sampling(pool_size, n_samples):
    return np.random.choice(pool_size, n_samples, replace=False)