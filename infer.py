# infer.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# 模型定义（需与训练时一致）
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
# 特征提取函数（保持与训练一致）
# -----------------------------
def process_features(df):
    onehot_fields = ['oracle_name', 'sut.component', 'sut.component_set', 'sut.module']
    codebert_dim = 768

    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(df[onehot_fields])

    tokenizer = AutoTokenizer.from_pretrained("./codebert")
    model = AutoModel.from_pretrained("./codebert").to("cpu")
    model.eval()

    def encode_column(column):
        embeddings = []
        with torch.no_grad():
            for text in column:
                text = str(text) if pd.notna(text) else ""
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.squeeze(0).numpy())
        return np.array(embeddings)

    api_ut_embeds = encode_column(df['api_ut'])
    tag_embeds = encode_column(df['tags'])

    X = np.hstack([onehot_encoded, api_ut_embeds, tag_embeds])
    return X

# -----------------------------
# 推理主流程
# -----------------------------
def infer(csv_path, model_path="log_classifier.pt"):
    print("📦 正在加载数据...")
    df = pd.read_csv(csv_path)
    X = process_features(df)

    input_dim = X.shape[1]
    model = LogClassifier(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("🔍 开始推理...")
    inputs = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()

    df['predicted_label'] = preds
    df['prob_bug'] = probs[:, 1].numpy()

    # 保存或打印
    output_path = "inference_results.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ 推理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    infer("your_infer_data.csv")  # 替换为你的实际推理数据路径
