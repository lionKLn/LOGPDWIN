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

#显示的设置当前设备,npu:5
torch.npu.set_device(5)


def calculate_class_weights(y):
    class_count = torch.bincount(y)
    total = len(y)
    weights = total / (2 * class_count)
    return weights.float()


# ⭐ 改造后的训练函数
def train_model_active(
    X_train,
    y_train,
    epochs=10,
    hidden_dim=128
):
    device = get_device()

    dataset = FeatureDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    #不再在代码中显示的分配，而是模型自动分配到npu:5
    #model = LogClassifier(X_train.shape[1], hidden_dim).to(device)
    model = LogClassifier(X_train.shape[1], hidden_dim)

    #class_weights = calculate_class_weights(y_train, device)
    class_weights = calculate_class_weights(y_train)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            #修改，不需要to(device)
            #X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss={total_loss:.4f}")

    return model


# ⭐ 新增：预测概率（主动学习核心）
def predict_proba(model, X):
    # device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        #X = X.to(device)
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)

    return probs[:, 1].cpu().numpy()