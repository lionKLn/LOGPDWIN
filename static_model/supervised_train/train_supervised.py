import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os

# 导入模型（确保model.py在可导入路径下）
from model import LogClassifier
from model import EarlyStopping

# 优先导入NPU支持
try:
    import torch_npu

    npu_available = torch.npu.is_available()
except ImportError:
    npu_available = False


# ----------------------------
# 数据集类（独立封装，方便外部复用）
# ----------------------------
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        """
        特征数据集类
        :param X: 特征张量，形状为 [样本数, 特征维度]
        :param y: 标签张量，形状为 [样本数]
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# 核心功能函数（支持外部调用）
# ----------------------------
def get_device():
    """获取可用设备（NPU优先，其次CPU）"""
    return torch.device("npu:5" if npu_available else "cpu")


def load_and_split_data(data_path, test_size=0.2, seed=42):
    """
    加载数据并划分训练/测试集（分层抽样）
    :param data_path: 数据文件路径（.pkl）
    :param test_size: 测试集占比
    :param seed: 随机种子（保证可复现）
    :return: X_train, X_test, y_train, y_test（均为torch张量）
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    data = pd.read_pickle(data_path)
    X = torch.tensor(data["merged_features"].tolist(), dtype=torch.float32)
    y = torch.tensor(data["false_positive"].tolist(), dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y  # 分层抽样，保证标签分布一致
    )

    # 打印数据分布（可选，外部调用时可注释）
    print(f"数据加载完成：")
    print(f"- 总样本数：{len(X)} | 训练集：{len(X_train)} | 测试集：{len(X_test)}")
    print(
        f"- 训练集0类占比：{torch.sum(y_train == 0) / len(y_train):.2%}，1类占比：{torch.sum(y_train == 1) / len(y_train):.2%}")
    print(
        f"- 测试集0类占比：{torch.sum(y_test == 0) / len(y_test):.2%}，1类占比：{torch.sum(y_test == 1) / len(y_test):.2%}")

    return X_train, X_test, y_train, y_test


def calculate_class_weights(y_train, device):
    """
    计算类别权重（解决类不平衡）
    :param y_train: 训练集标签
    :param device: 目标设备（torch.device）
    :return: 类别权重张量（已移至目标设备）
    """
    class_count = torch.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (2 * class_count)  # 平衡权重公式
    return class_weights.float().to(device)


def train_model(
        data_path,
        save_model_path="best_log_classifier.pt",
        batch_size=32,
        epochs=50,
        learning_rate=5e-4,
        hidden_dim=128,
        test_size=0.2,
        random_seed=42,
        pos_label=0  # 核心关注的正类（0类=非误报）
):
    """
    模型训练主函数（支持外部模块调用）
    :param data_path: 数据文件路径
    :param save_model_path: 模型保存路径
    :param batch_size: 批次大小
    :param epochs: 训练轮数
    :param learning_rate: 学习率
    :param hidden_dim: 模型隐藏层维度
    :param test_size: 测试集占比
    :param random_seed: 随机种子
    :param pos_label: 正类标签（默认0类）
    :return: 训练完成的模型、最终测试集指标字典
    """
    # 1. 初始化设备
    device = get_device()
    print(f"使用设备：{device}")

    # 2. 加载并划分数据
    X_train, X_test, y_train, y_test = load_and_split_data(
        data_path, test_size=test_size, seed=random_seed
    )

    # 3. 创建数据加载器
    train_dataset = FeatureDataset(X_train, y_train)
    test_dataset = FeatureDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 4. 初始化模型、损失函数、优化器
    input_dim = X_train.shape[1]
    model = LogClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    class_weights = calculate_class_weights(y_train, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    # 5. 早停机制
    early_stopping = EarlyStopping(
        patience=5,
        verbose=True,
        delta=0.001,
        path=save_model_path
    )

    # 6. 训练循环
    print(f"\n开始训练（核心关注：{pos_label}类）...")
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds, train_true = [], []

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_true.extend(batch_y.cpu().numpy())

        # 计算训练集指标
        train_avg_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_true, train_preds)
        train_precision = precision_score(train_true, train_preds, average="binary", pos_label=pos_label)
        train_recall = recall_score(train_true, train_preds, average="binary", pos_label=pos_label)
        train_f1 = f1_score(train_true, train_preds, average="binary", pos_label=pos_label)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        # 计算验证集指标
        val_avg_loss = val_loss / len(test_loader.dataset)
        val_acc = accuracy_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds, average="binary", pos_label=pos_label)
        val_recall = recall_score(val_true, val_preds, average="binary", pos_label=pos_label)
        val_f1 = f1_score(val_true, val_preds, average="binary", pos_label=pos_label)

        # 打印日志
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_avg_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")

        # 学习率衰减 + 早停判断
        scheduler.step()
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("早停触发，训练结束！")
            break

    # 7. 加载最优模型并评估
    best_model = LogClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    best_model.load_state_dict(torch.load(save_model_path, map_location=device))
    best_model.eval()

    # 最终测试集评估
    final_preds, final_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = best_model(batch_x)
            final_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            final_true.extend(batch_y.numpy())

    # 计算最终指标（含0类和1类）
    final_metrics = {
        "overall": {
            "accuracy": accuracy_score(final_true, final_preds)
        },
        f"class_{pos_label}": {  # 核心关注的类（0类）
            "precision": precision_score(final_true, final_preds, average="binary", pos_label=pos_label),
            "recall": recall_score(final_true, final_preds, average="binary", pos_label=pos_label),
            "f1": f1_score(final_true, final_preds, average="binary", pos_label=pos_label)
        },
        f"class_{1 - pos_label}": {  # 对比类（1类）
            "precision": precision_score(final_true, final_preds, average="binary", pos_label=1 - pos_label),
            "recall": recall_score(final_true, final_preds, average="binary", pos_label=1 - pos_label),
            "f1": f1_score(final_true, final_preds, average="binary", pos_label=1 - pos_label)
        }
    }

    # 打印最终结果
    print(f"\n==================== 最终测试集性能 ====================")
    print(f"整体准确率：{final_metrics['overall']['accuracy']:.4f}")
    print(f"\n核心类（{pos_label}类）指标：")
    print(f"精确率：{final_metrics[f'class_{pos_label}']['precision']:.4f}")
    print(f"召回率：{final_metrics[f'class_{pos_label}']['recall']:.4f}")
    print(f"F1分数：{final_metrics[f'class_{pos_label}']['f1']:.4f}")
    print(f"\n对比类（{1 - pos_label}类）指标：")
    print(f"精确率：{final_metrics[f'class_{1 - pos_label}']['precision']:.4f}")
    print(f"召回率：{final_metrics[f'class_{1 - pos_label}']['recall']:.4f}")
    print(f"F1分数：{final_metrics[f'class_{1 - pos_label}']['f1']:.4f}")
    print("========================================================")

    # 返回模型和指标（方便外部调用时进一步处理）
    return best_model, final_metrics


# ----------------------------
# 主函数（用于直接运行脚本）
# ----------------------------
if __name__ == "__main__":
    # 可在此处修改默认参数
    model, metrics = train_model(
        data_path="processed_dataset.pkl",
        save_model_path="best_log_classifier.pt",
        batch_size=32,
        epochs=50,
        pos_label=0  # 核心关注0类
    )
    # 示例：外部调用时可获取模型和指标进行后续操作
    print("\n训练完成，模型和指标已返回")