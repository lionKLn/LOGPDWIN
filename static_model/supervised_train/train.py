import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
# 导入你的模型
from model import LogClassifier
from model import EarlyStopping

# 优先导入NPU支持
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except ImportError:
    npu_available = False

# ----------------------------
# 1. 配置参数（新增：明确正类为0）
# ----------------------------
DEVICE = torch.device("npu:5" if npu_available else "cpu")
DATA_PATH = "processed_dataset.pkl"
SAVE_MODEL_PATH = "best_log_classifier.pt"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-4
HIDDEN_DIM = 128
TEST_SIZE = 0.2
RANDOM_SEED = 42
POS_LABEL = 0  # 核心修改：指定正类为0（非误报），所有指标围绕0类计算


# ----------------------------
# 2. 加载数据并划分训练/测试集（无修改）
# ----------------------------
def load_and_split_data(data_path, test_size=0.2, seed=42):
    data = pd.read_pickle(data_path)
    X = torch.tensor(data["merged_features"].tolist(), dtype=torch.float32)
    y = torch.tensor(data["false_positive"].tolist(), dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y  # 分层抽样，保证0类在训练/测试集分布一致
    )

    print(f"数据加载完成（核心关注：0类=非误报）：")
    print(f"- 总样本数：{len(X)} | 训练集：{len(X_train)} | 测试集：{len(X_test)}")
    print(
        f"- 训练集标签分布：0类（非误报）占比 {torch.sum(y_train == 0) / len(y_train):.2%}，1类（误报）占比 {torch.sum(y_train == 1) / len(y_train):.2%}")
    print(
        f"- 测试集标签分布：0类（非误报）占比 {torch.sum(y_test == 0) / len(y_test):.2%}，1类（误报）占比 {torch.sum(y_test == 1) / len(y_test):.2%}")

    return X_train, X_test, y_train, y_test


# ----------------------------
# 3. 定义数据集类（无修改）
# ----------------------------
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# 4. 计算类别权重（无修改，但需注意：0类是少数类，权重会更大）
# ----------------------------
def calculate_class_weights(y_train):
    class_count = torch.bincount(y_train)  # [0类数量, 1类数量]
    total_samples = len(y_train)
    class_weights = total_samples / (2 * class_count)  # 0类（少数）权重更大，符合关注0类需求
    class_weights = class_weights.float().to(DEVICE)
    print(f"类别权重（0类=非误报，权重更大以优先学习）：0类={class_weights[0]:.4f}，1类={class_weights[1]:.4f}")
    return class_weights


# ----------------------------
# 5. 模型训练与验证（核心修改：围绕0类计算指标，早停监控0类F1）
# ----------------------------
def train_model():
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, TEST_SIZE, RANDOM_SEED)

    train_dataset = FeatureDataset(X_train, y_train)
    test_dataset = FeatureDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 初始化模型（类别权重已倾向0类）
    input_dim = X_train.shape[1]
    model = LogClassifier(input_dim=input_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
    class_weights = calculate_class_weights(y_train)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)  # 加权损失优先优化0类预测
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    # 早停：监控0类的F1分数（核心关注0类，分数越高越好）
    early_stopping = EarlyStopping(
        patience=5,
        verbose=True,
        delta=0.001,
        path=SAVE_MODEL_PATH
    )

    print(f"\n开始训练（设备：{DEVICE}，核心关注：0类=非误报的预测效果）...")
    for epoch in range(1, EPOCHS + 1):
        # ---------------------- 训练阶段（围绕0类计算指标） ----------------------
        model.train()
        train_loss = 0.0
        train_preds = []
        train_true = []

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_true.extend(batch_y.cpu().numpy())

        # 计算训练集指标：所有指标围绕0类（POS_LABEL=0）
        train_avg_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_true, train_preds)
        # 关键：pos_label=0，指标反映0类的预测效果
        train_precision_0 = precision_score(train_true, train_preds, average="binary", pos_label=POS_LABEL)
        train_recall_0 = recall_score(train_true, train_preds, average="binary", pos_label=POS_LABEL)
        train_f1_0 = f1_score(train_true, train_preds, average="binary", pos_label=POS_LABEL)

        # ---------------------- 验证阶段（围绕0类计算指标） ----------------------
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        # 计算验证集指标：围绕0类
        val_avg_loss = val_loss / len(test_loader.dataset)
        val_acc = accuracy_score(val_true, val_preds)
        val_precision_0 = precision_score(val_true, val_preds, average="binary", pos_label=POS_LABEL)
        val_recall_0 = recall_score(val_true, val_preds, average="binary", pos_label=POS_LABEL)
        val_f1_0 = f1_score(val_true, val_preds, average="binary", pos_label=POS_LABEL)

        # ---------------------- 日志打印（突出0类指标） ----------------------
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Train_0类（非误报）: Prec={train_precision_0:.4f}, Rec={train_recall_0:.4f}, F1={train_f1_0:.4f} | "
              f"Val Loss: {val_avg_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val_0类（非误报）: Prec={val_precision_0:.4f}, Rec={val_recall_0:.4f}, F1={val_f1_0:.4f}")

        # 学习率衰减
        scheduler.step()

        # 早停判断：监控0类的F1分数（核心！确保模型优先优化0类预测）
        early_stopping(val_f1_0, model)  # 无需加负号：0类F1越高越好，符合早停类逻辑
        if early_stopping.early_stop:
            print("早停触发（0类F1连续5轮无提升），训练结束！")
            break

    # ---------------------- 最终测试集评估（突出0类效果） ----------------------
    print(f"\n加载最优模型（{SAVE_MODEL_PATH}）...")
    best_model = LogClassifier(input_dim=input_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
    best_model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))

    best_model.eval()
    final_preds = []
    final_true = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = best_model(batch_x)
            final_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            final_true.extend(batch_y.numpy())

    # 计算最终指标：重点展示0类的精确率、召回率、F1
    final_acc = accuracy_score(final_true, final_preds)
    final_precision_0 = precision_score(final_true, final_preds, average="binary", pos_label=POS_LABEL)
    final_recall_0 = recall_score(final_true, final_preds, average="binary", pos_label=POS_LABEL)
    final_f1_0 = f1_score(final_true, final_preds, average="binary", pos_label=POS_LABEL)
    # 可选：展示1类指标作为对比
    final_precision_1 = precision_score(final_true, final_preds, average="binary", pos_label=1)
    final_recall_1 = recall_score(final_true, final_preds, average="binary", pos_label=1)
    final_f1_1 = f1_score(final_true, final_preds, average="binary", pos_label=1)

    # 日志：突出0类指标的业务意义
    print(f"\n==================== 最终测试集性能（核心关注：0类=非误报） ====================")
    print(f"整体准确率（Accuracy）: {final_acc:.4f}")
    print(f"\n【0类（非误报）核心指标】")
    print(f"精确率（Precision）: {final_precision_0:.4f} → 预测为非误报的样本中，实际是non-误报的比例（避免误判正常样本）")
    print(f"召回率（Recall）: {final_recall_0:.4f} → 实际是非误报的样本中，被正确预测的比例（避免漏判正常样本）")
    print(f"F1分数: {final_f1_0:.4f} → 0类预测效果的综合评价")
    print(f"\n【1类（误报）对比指标】")
    print(f"精确率: {final_precision_1:.4f}, 召回率: {final_recall_1:.4f}, F1: {final_f1_1:.4f}")
    print("=============================================================================")


# ----------------------------
# 6. 启动训练
# ----------------------------
if __name__ == "__main__":
    train_model()