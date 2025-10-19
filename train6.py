import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, f1_score, precision_score  # 统一导入所有指标函数
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import os

# -----------------------------
# 设备配置
# -----------------------------
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except ImportError:
    npu_available = False

device = torch.device("npu:6" if npu_available else "cuda:0" if torch.cuda.is_available() else "cpu")
if npu_available:
    torch.npu.set_device(0)

print(f"✅ 使用设备: {device}")

# -----------------------------
# 特征提取
# -----------------------------
def process_features(df):
    onehot_fields = ['oracle_name', 'sut.component', 'sut.component_set', 'sut.module']
    codebert_dim = 768

    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(df[onehot_fields])

    tokenizer = AutoTokenizer.from_pretrained("./codebert")
    model = AutoModel.from_pretrained("./codebert").to(device)
    model.eval()

    def encode_column(column):
        embeddings = []
        with torch.no_grad():
            for text in column:
                text = str(text) if pd.notna(text) else ""
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :] if outputs.last_hidden_state.size(1) > 0 else torch.zeros(1, codebert_dim).to(device)
                embeddings.append(cls_embedding.squeeze(0).cpu().numpy())
        return np.array(embeddings)

    api_ut_embeds = encode_column(df['api_ut'])
    tag_embeds = encode_column(df['tags'])
    X = np.hstack([onehot_encoded, api_ut_embeds, tag_embeds])
    y = df['false_positives'].astype(int).values
    return X, y

# -----------------------------
# Dataset 和 DataLoader
# -----------------------------
class LogDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(X, y, batch_size=32, shuffle=True):
    dataset = LogDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# -----------------------------
# 模型定义
# -----------------------------
class LogClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# 主流程
# -----------------------------
if __name__ == "__main__":
    csv_path = "your_data.csv"
    df = pd.read_csv(csv_path)

    print("🚀 正在提取特征...")
    X, y = process_features(df)

    # 划分数据集（分层抽样，保证0/1类分布一致）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # 打印数据分布，明确0/1类占比（新增）
    print(f"📊 数据分布：")
    print(f"- 训练集：0类（非误报）{np.sum(y_train==0)}个，1类（误报）{np.sum(y_train==1)}个")
    print(f"- 测试集：0类（非误报）{np.sum(y_test==0)}个，1类（误报）{np.sum(y_test==1)}个")

    print("🔁 正在对训练集上采样以缓解类别不平衡...")
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    print(f"- 上采样后训练集：0类{np.sum(y_resampled==0)}个，1类{np.sum(y_resampled==1)}个（平衡）")

    batch_size = 32
    train_loader = get_dataloader(X_resampled, y_resampled, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    input_dim = X_resampled.shape[1]
    model = LogClassifier(input_dim).to(device)

    # 计算类别权重（上采样后0/1类平衡，权重接近1:1）
    class_counts = Counter(y_resampled)
    weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"⚖️  类别权重：0类={weights[0]:.4f}，1类={weights[1]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 初始化 TensorBoard 日志
    log_dir = f"runs/log_classifier_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for features, labels in tqdm(train_loader, desc=f"🟢 Epoch {epoch + 1}/{epochs}"):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # ---------------------- 训练集指标：同时计算1类和0类（新增0类指标） ----------------------
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        # 1类（误报）指标（显式指定pos_label=1）
        train_recall_1 = recall_score(all_labels, all_preds, average="binary", pos_label=1)
        train_precision_1 = precision_score(all_labels, all_preds, average="binary", pos_label=1)
        train_f1_1 = f1_score(all_labels, all_preds, average="binary", pos_label=1)
        # 0类（非误报，核心关注）指标（显式指定pos_label=0）
        train_recall_0 = recall_score(all_labels, all_preds, average="binary", pos_label=0)
        train_precision_0 = precision_score(all_labels, all_preds, average="binary", pos_label=0)
        train_f1_0 = f1_score(all_labels, all_preds, average="binary", pos_label=0)

        # 打印训练日志：突出0类指标（核心目标）
        print(f"📘 Epoch {epoch + 1} - Loss: {total_loss:.4f} - Acc: {acc * 100:.2f}%")
        print(f"   ├─ 1类（误报）: Precision={train_precision_1:.2f}, Recall={train_recall_1:.2f}, F1={train_f1_1:.2f}")
        print(f"   └─ 0类（非误报，核心）: Precision={train_precision_0:.2f}, Recall={train_recall_0:.2f}, F1={train_f1_0:.2f}")

        # TensorBoard 写入：同时记录1类和0类指标（新增0类日志）
        writer.add_scalar("Train/Loss", total_loss, epoch)
        writer.add_scalar("Train/Accuracy", acc, epoch)
        # 1类指标
        writer.add_scalar("Train/1类_Recall", train_recall_1, epoch)
        writer.add_scalar("Train/1类_Precision", train_precision_1, epoch)
        writer.add_scalar("Train/1类_F1", train_f1_1, epoch)
        # 0类指标（核心）
        writer.add_scalar("Train/0类_Recall", train_recall_0, epoch)
        writer.add_scalar("Train/0类_Precision", train_precision_0, epoch)
        writer.add_scalar("Train/0类_F1", train_f1_0, epoch)

    # ---------------------- 测试集评估：同时计算1类和0类（新增0类指标） ----------------------
    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs, 1)

            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # 计算测试集指标
    val_acc = (np.array(val_preds) == np.array(val_labels)).mean()
    # 1类（误报）指标
    val_recall_1 = recall_score(val_labels, val_preds, average="binary", pos_label=1)
    val_precision_1 = precision_score(val_labels, val_preds, average="binary", pos_label=1)
    val_f1_1 = f1_score(val_labels, val_preds, average="binary", pos_label=1)
    # 0类（非误报，核心关注）指标
    val_recall_0 = recall_score(val_labels, val_preds, average="binary", pos_label=0)
    val_precision_0 = precision_score(val_labels, val_preds, average="binary", pos_label=0)
    val_f1_0 = f1_score(val_labels, val_preds, average="binary", pos_label=0)

    # 打印测试日志：突出0类指标（核心目标）
    print(f"\n🧪 测试集最终评估（核心关注：0类=非误报，真正有问题的数据）")
    print(f"📊 整体性能：Acc={val_acc * 100:.2f}%")
    print(f"🔴 1类（误报）性能：")
    print(f"   ├─ Precision: {val_precision_1:.2f}（预测为误报的样本中，实际是误报的比例）")
    print(f"   ├─ Recall:    {val_recall_1:.2f}（实际是误报的样本中，被正确识别的比例）")
    print(f"   └─ F1:        {val_f1_1:.2f}")
    print(f"🟢 0类（非误报，核心）性能：")
    print(f"   ├─ Precision: {val_precision_0:.2f}（预测为有问题的样本中，实际有问题的比例→推荐可靠性）")
    print(f"   ├─ Recall:    {val_recall_0:.2f}（实际有问题的样本中，被正确识别的比例→是否漏推荐）")
    print(f"   └─ F1:        {val_f1_0:.2f}（推荐效果综合评价）")

    # TensorBoard 写入测试集指标（含0类）
    writer.add_scalar("Test/Accuracy", val_acc, epochs)
    # 1类指标
    writer.add_scalar("Test/1类_Recall", val_recall_1, epochs)
    writer.add_scalar("Test/1类_Precision", val_precision_1, epochs)
    writer.add_scalar("Test/1类_F1", val_f1_1, epochs)
    # 0类指标（核心）
    writer.add_scalar("Test/0类_Recall", val_recall_0, epochs)
    writer.add_scalar("Test/0类_Precision", val_precision_0, epochs)
    writer.add_scalar("Test/0类_F1", val_f1_0, epochs)
    writer.close()

    # 保存模型
    torch.save(model.state_dict(), "log_classifier.pt")
    print(f"\n✅ 模型已保存为 log_classifier.pt")

    # 过拟合判断：同时参考1类和0类指标
    print(f"\n🔍 过拟合判断建议")
    print(f"   ├─ 0类F1差异：Train F1={train_f1_0:.2f} vs Test F1={val_f1_0:.2f} → Δ={train_f1_0 - val_f1_0:.4f}")
    print(f"   ├─ 1类F1差异：Train F1={train_f1_1:.2f} vs Test F1={val_f1_1:.2f} → Δ={train_f1_1 - val_f1_1:.4f}")
    print(f"   └─ 建议：若Δ>0.1，说明训练指标远高于测试指标，可能存在过拟合（优先关注0类差异）")