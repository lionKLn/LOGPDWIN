import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel

# 0. 设备检测：优先 NPU，其次 CUDA，否则 CPU
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except ImportError:
    npu_available = False

if npu_available:
    device = torch.device("npu:0")
    # 可选：设定默认 NPU
    torch.npu.set_device(0)
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 1. 加载 CSV
data = pd.read_csv('dataset/labeled_data.csv')

# 2. 分离特征和标签
y = (data['false_positives'] == 'FALSE').astype(int).values  # 1 表示真正异常

# 3. One-Hot 编码其他列
categorical_cols = ['api_ut', 'oracle_name', 'sut.component', 'sut.component_set', 'sut.module']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(data[categorical_cols].astype(str))

# 4. CodeBERT 嵌入 tags
tokenizer = AutoTokenizer.from_pretrained("./codebert")
codebert = AutoModel.from_pretrained("./codebert").to(device)
codebert.eval()  # 仅用于 embedding

def embed_codebert(text_list, batch_size=16):
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            # 把输入张量搬到 device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = codebert(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, 768)
            all_emb.append(cls_emb.cpu())  # 收集到 CPU，以便后续 concat/numpy
    return torch.cat(all_emb, dim=0)

tags = data['tags'].fillna("").tolist()
X_tags = embed_codebert(tags)  # torch.Tensor on CPU now

# 5. 拼接所有特征向量
# X_cat 是 numpy，X_tags 是 torch.Tensor.cpu()
X = np.concatenate([X_cat, X_tags.numpy()], axis=1)
print(f"特征总维度：{X.shape}")

# 6. 划分训练/测试集
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 转为张量并搬到 device
X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32, device=device)
y_test  = torch.tensor(y_test_np,  dtype=torch.long, device=device)

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

model = LogClassifier(X_train.size(1)).to(device)

# 损失 + 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 8. 训练循环
epochs = 15
for epoch in range(1, epochs + 1):
    model.train()
    logits = model(X_train)      # X_train 已在 device
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

# 10. 测试评估之后，保存 state_dict
save_path = "log_classifier_state_dict.pth"
torch.save(model.state_dict(), save_path)
print(f"Model state_dict saved to {save_path}")