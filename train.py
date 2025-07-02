import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# 0. 设备检测：优先 NPU，其次 CUDA，否则 CPU
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except ImportError:
    npu_available = False

if npu_available:
    device = torch.device("npu:0")
    torch.npu.set_device(0)
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 1. 加载 CSV
data = pd.read_csv('dataset/labeled_data.csv')

# 2. 分离标签（1 表示真正异常）
y = (data['false_positives'] == 'FALSE').astype(int).values

# 3. One-Hot 编码其他列
categorical_cols = ['api_ut', 'oracle_name', 'sut.component', 'sut.component_set', 'sut.module']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(data[categorical_cols].astype(str))

# 4. CodeBERT 嵌入 tags
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
codebert.eval()

def embed_codebert(texts, batch_size=16):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(batch,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=128)
            tokens = {k:v.to(device) for k,v in tokens.items()}
            out = codebert(**tokens)
            cls_emb = out.last_hidden_state[:,0,:]  # (B, 768)
            embs.append(cls_emb.cpu())
    return torch.cat(embs, dim=0)

tags = data['tags'].fillna("").tolist()
X_tags = embed_codebert(tags)  # CPU tensor

# 5. 拼接所有特征并划分训练/验证/测试
X_all = np.concatenate([X_cat, X_tags.numpy()], axis=1)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val
)

# 6. 转为 TensorDataset + DataLoader
def make_loader(X, y, batch_size=32, shuffle=False):
    tX = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(tX, ty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, batch_size=64, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   batch_size=64, shuffle=False)
test_loader  = make_loader(X_test,  y_test,  batch_size=64, shuffle=False)

# 7. 定义模型
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

model = LogClassifier(input_dim=X_all.shape[1], hidden_dim=256).to(device)

# 8. 损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 9. 训练 + 验证
best_val_f1 = 0.0
best_model_path = "outputs/best_model.pth"
os.makedirs("outputs", exist_ok=True)

for epoch in range(1, 16):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)

    # 验证
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(yb.cpu().tolist())
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch:02d} — train_loss: {avg_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {f1:.4f}")

    # 保存最优
    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(model.state_dict(), best_model_path)

# 10. 测试集评估
print("\n=== Testing Best Model ===")
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(yb.cpu().tolist())

print(classification_report(all_labels, all_preds, target_names=["FP", "TrueAnomaly"]))