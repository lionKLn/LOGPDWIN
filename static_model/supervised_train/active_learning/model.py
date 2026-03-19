import torch
import torch.nn as nn


class LogClassifier(nn.Module):
    """
    简单单层分类器（含1个隐藏层的浅层神经网络）
    用于二分类任务（如：误报/非误报）
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        """
        参数：
            input_dim: 输入特征的维度（融合后的特征总维度）
            hidden_dim: 隐藏层维度（默认128，可根据数据调整）
            dropout: Dropout概率（默认0.2，用于防止过拟合）
        """
        super().__init__()
        self.net = nn.Sequential(
            # 第一层：高维特征映射到隐藏层
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # 引入非线性
            nn.Dropout(dropout),  # 随机失活部分神经元，防止过拟合
            # 第二层：隐藏层映射到二分类输出
            nn.Linear(hidden_dim, 2)  # 输出维度=2（对应0/1两类）
        )

    def forward(self, x):
        """
        前向传播：将输入特征映射到分类结果
        参数：
            x: 输入特征张量，形状为 [batch_size, input_dim]
        返回：
            未归一化的分类得分，形状为 [batch_size, 2]
        """
        # 确保输入是二维张量（batch_size, input_dim），兼容单样本输入
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 单样本时增加batch维度
        return self.net(x)


class EarlyStopping:
    """
    早停机制类：监控验证集指标（如F1分数），当指标不再提升时停止训练
    适用于“指标越高越好”的场景（如准确率、F1分数）
    """
    def __init__(self, patience=5, verbose=False, delta=0.0, path='best_model.pt'):
        """
        参数：
            patience: 容忍指标不提升的最大epoch数（默认5）
            verbose: 是否打印早停日志（默认False）
            delta: 指标最小提升阈值（小于该值视为无提升，默认0.0）
            path: 最优模型保存路径（默认'best_model.pt'）
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0  # 记录指标未提升的连续epoch数
        self.best_score = None  # 最佳指标分数
        self.early_stop = False  # 是否触发早停

    def __call__(self, current_score, model):
        """
        每轮验证后调用，判断是否更新最佳模型或触发早停
        参数：
            current_score: 当前epoch的验证集指标（如F1分数）
            model: 当前训练的模型
        """
        # 初始化最佳分数（第一轮直接保存）
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(current_score, model)
        # 若当前分数未超过最佳分数+delta（无显著提升）
        elif current_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"早停计数器: {self.counter} / {self.patience}")
            # 连续patience轮无提升，触发早停
            if self.counter >= self.patience:
                self.early_stop = True
        # 若当前分数超过最佳分数+delta（有显著提升）
        else:
            self.best_score = current_score
            self._save_checkpoint(current_score, model)
            self.counter = 0  # 重置计数器

    def _save_checkpoint(self, score, model):
        """保存当前最优模型"""
        if self.verbose:
            print(f"验证集指标提升 ({self.best_score:.4f} → {score:.4f})，保存模型至 {self.path}")
        # 保存模型参数（仅保存状态字典，节省空间）
        torch.save(model.state_dict(), self.path)