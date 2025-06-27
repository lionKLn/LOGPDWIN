import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel

# 1. 加载 CSV
data = pd.read_csv('dataset/data1.csv')

# 2. 分离特征和标签
y = (data['false_positives'] == 'FALSE').astype(int).values  # 1 表示真正异常

# 3. One‑Hot 编码其他列
categorical_cols = ['api_ut', 'oracle_name', 'component', 'component_set', 'module']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(data[categorical_cols].astype(str))

# 4. CodeBERT 嵌入 tags
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert = AutoModel.from_pretrained("microsoft/codebert-base")
codebert.eval()  # 仅用于 embedding

def embed_codebert(text_list, batch_size=16):
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = codebert(**inputs)
            cls_emb = outputs.last_hidden_state[:,0,:]
            all_emb.append(cls_emb)
    return torch.cat(all_emb, dim=0)

tags = data['tags'].fillna("").tolist()
X_tags = embed_codebert(tags)

# 5. 拼接所有特征向量
X = np.concatenate([X_cat, X_tags.numpy()], axis=1)
print(f"特征总维度：{X.shape}")

# 6. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 7. 建立简单全连接模型
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

model = LogClassifier(X_train.size(1))
device = torch.device("cpu")
model.to(device)

# 损失 + 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 8. 训练循环
epochs = 15
for epoch in range(1, epochs+1):
    model.train()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

# 9. 测试评估
model.eval()
with torch.no_grad():
    logits = model(X_test)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == y_test).float().mean().item()
    print(f"Test Accuracy: {acc*100:.2f}%")
