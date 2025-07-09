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

# -----------------------------
# 设备配置（NPU 优先，其次 CUDA）
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
# 特征提取函数
# -----------------------------
def process_features(df):
    onehot_fields = ['oracle_name', 'sut.component', 'sut.component_set', 'sut.module']
    codebert_fields = ['api_ut', 'tags']
    codebert_dim = 768

    # 1. One-Hot 编码
    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(df[onehot_fields])

    # 2. 加载 CodeBERT
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
                if outputs.last_hidden_state.size(1) == 0:
                    cls_embedding = torch.zeros(1, codebert_dim).to(device)
                else:
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.squeeze(0).cpu().numpy())
        return np.array(embeddings)

    # 3. 提取 CodeBERT 特征
    api_ut_embeds = encode_column(df['api_ut'])
    tag_embeds = encode_column(df['tags'])

    # 4. 拼接所有特征
    X = np.hstack([onehot_encoded, api_ut_embeds, tag_embeds])
    y = df['false_positives'].astype(int).values
    return X, y

# -----------------------------
# 自定义 Dataset 类
# -----------------------------
class LogDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_feature_dim(self):
        return self.X.shape[1]

# -----------------------------
# Dataloader 构造函数
# -----------------------------
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
    # Step 1: 读取和预处理数据
    csv_path = "your_data.csv"  # 替换为你的路径
    df = pd.read_csv(csv_path)

    print("🚀 正在提取特征...")
    X, y = process_features(df)

    # Step 2: 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Step 3: 构建 Dataloader
    batch_size = 32
    train_loader = get_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    # Step 4: 初始化模型
    input_dim = X_train.shape[1]
    model = LogClassifier(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Step 5: 训练模型
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        all_preds, all_labels = [], []

        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

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
        f1 = f1_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - "
              f"Train Acc: {acc*100:.2f}% - Recall: {recall:.2f} - F1: {f1:.2f}")

    # Step 6: 测试集评估
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
    val_f1 = f1_score(val_labels, val_preds)

    print(f"\n🧪 Test Set Evaluation - Acc: {val_acc*100:.2f}% - Recall: {val_recall:.2f} - F1: {val_f1:.2f}")

    # Step 7: 保存模型
    torch.save(model.state_dict(), "log_classifier.pt")
    print("✅ 模型已保存为 log_classifier.pt")
