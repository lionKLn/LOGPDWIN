import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, f1_score
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import precision_score
import os
import joblib  # ✅ 新增保存编码器用

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
def process_features(df, save_encoder=True):
    onehot_fields = ['oracle_name', 'sut.component', 'sut.component_set', 'sut.module']
    codebert_dim = 768

    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(df[onehot_fields])

    if save_encoder:
        joblib.dump(onehot_encoder, 'encoder.pkl')
        np.save('encoder_columns.npy', onehot_encoder.get_feature_names_out())

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("🔁 正在对训练集上采样以缓解类别不平衡...")
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    print("已划分数据集")

    batch_size = 32
    train_loader = get_dataloader(X_resampled, y_resampled, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    input_dim = X_resampled.shape[1]
    model = LogClassifier(input_dim).to(device)

    class_counts = Counter(y_resampled)
    weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

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

        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print(f"📘 Epoch {epoch + 1} - Loss: {total_loss:.4f} - Acc: {acc * 100:.2f}% - Precision: {precision:.2f} - Recall: {recall:.2f} - F1: {f1:.2f}")

        writer.add_scalar("Train/Loss", total_loss, epoch)
        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Train/Recall", recall, epoch)
        writer.add_scalar("Train/Precision", precision, epoch)
        writer.add_scalar("Train/F1", f1, epoch)

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

    val_acc = (np.array(val_preds) == np.array(val_labels)).mean()
    val_recall = recall_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)

    print(f"\n🧪 Test Evaluation")
    print(f"Acc:     {val_acc * 100:.2f}%")
    print(f"Recall:  {val_recall:.2f}")
    print(f"Precision: {val_precision:.2f}")
    print(f"F1:      {val_f1:.2f}")

    writer.add_scalar("Test/Accuracy", val_acc, epochs)
    writer.add_scalar("Test/Recall", val_recall, epochs)
    writer.add_scalar("Test/Precision", val_precision, epochs)
    writer.add_scalar("Test/F1", val_f1, epochs)
    writer.close()

    torch.save(model.state_dict(), "log_classifier.pt")
    print("✅ 模型已保存为 log_classifier.pt")

    print("\n🔍 过拟合判断建议（参考指标差异）")
    print(f"Train F1 vs Test F1: Δ = {f1 - val_f1:.4f}")
    print(f"Train Precision vs Test Precision: Δ = {precision - val_precision:.4f}")
    print("建议：若训练指标远高于测试指标，可能存在过拟合现象。")
