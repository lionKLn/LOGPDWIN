import torch
from torch.utils.data import Dataset, DataLoader
from model import LogClassifier


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_device():
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device("npu:5")
    except:
        pass
    return torch.device("cpu")


def calculate_class_weights(y):
    class_count = torch.bincount(y)
    total = len(y)
    weights = total / (2 * class_count)
    return weights.float()


# ⭐ 改造后的训练函数
def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=20,
    batch_size=32,
    lr=5e-4,
    hidden_dim=128
):
    device = get_device()

    train_dataset = FeatureDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    model = LogClassifier(input_dim, hidden_dim).to(device)

    class_weights = calculate_class_weights(y_train).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss:.4f}")

    return model


# ⭐ 新增：预测概率（主动学习核心）
def predict_proba(model, X):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)

    return probs[:, 1].cpu().numpy()