import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
from dataloader_module import LogDataset, get_dataloader

# Step 4: 初始化模型、损失函数、优化器
# 如果安装并启用了 torch_npu，就可以用 NPU
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except ImportError:
    npu_available = False

# 再判断 CUDA
cuda_available = torch.cuda.is_available()

# 优先 NPU，其次 CUDA，否则 CPU
if npu_available:
    device = torch.device("npu:6")
    # （可选）设定当前进程使用的 NPU 号
    torch.npu.set_device(0)
elif cuda_available:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
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
csv_path = "your_data.csv"  # 替换为你的CSV路径
batch_size = 32

# Step 1: 加载并划分数据
df = pd.read_csv(csv_path)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Step 2: 构建 DataLoader
train_loader = get_dataloader(train_df, batch_size=batch_size, shuffle=True)
test_loader = get_dataloader(test_df, batch_size=batch_size, shuffle=False)

# Step 3: 获取输入维度
train_dataset = LogDataset(train_df)
input_dim = train_dataset.get_feature_dim()


model = LogClassifier(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Step 5: 训练模型
epochs = 5
for epoch in range(epochs):
    total_loss = 0.0
    all_preds = []
    all_labels = []

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

    # 训练集指标
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - "
          f"Train Acc: {acc*100:.2f}% - Recall: {recall:.2f} - F1: {f1:.2f}")

# Step 6: 在测试集上验证
model.eval()
val_preds = []
val_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

        val_preds.extend(predicted.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

val_acc = (torch.tensor(val_preds) == torch.tensor(val_labels)).sum().item() / len(val_labels)
val_recall = recall_score(val_labels, val_preds)
val_f1 = f1_score(val_labels, val_preds)

print(f"\n🧪 Test Set Evaluation - Acc: {val_acc*100:.2f}% - Recall: {val_recall:.2f} - F1: {val_f1:.2f}")

# Step 7: 保存模型
torch.save(model.state_dict(), "log_classifier.pt")
print("✅ 模型已保存为 log_classifier.pt")
