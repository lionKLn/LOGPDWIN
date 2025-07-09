import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, f1_score
from dataloader_module import LogDataset, get_dataloader

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


# Step 1: 加载数据
csv_path = "your_data.csv"  # 替换为你的CSV路径
batch_size = 32
train_loader = get_dataloader(csv_path, batch_size=batch_size, shuffle=True)

# Step 2: 获取输入维度
dataset = LogDataset(csv_path)
input_dim = dataset.get_feature_dim()

# Step 3: 初始化模型、损失函数、优化器
device = torch.device("npu" if hasattr(torch, 'npu') and torch.npu.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")


model = LogClassifier(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Step 4: 训练模型
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

    # 计算指标
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - "
          f"Acc: {acc*100:.2f}% - Recall: {recall:.2f} - F1: {f1:.2f}")

# Step 5: 保存模型
torch.save(model.state_dict(), "log_classifier.pt")
print("✅ 模型已保存为 log_classifier.pt")
