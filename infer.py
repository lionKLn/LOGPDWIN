# infer.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 模型结构保持一致
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
# 特征处理（推理版）
# -----------------------------
def process_features_infer(df, encoder, encoder_columns, device):
    onehot_fields = ['oracle_name', 'sut.component', 'sut.component_set', 'sut.module']
    codebert_dim = 768

    # 确保缺失输入列补齐（原始字段）
    for col in onehot_fields:
        if col not in df.columns:
            df[col] = ""

    # OneHot 编码（维度和 encoder.pkl 一致）
    onehot_encoded = encoder.transform(df[onehot_fields])
    onehot_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out())

    # 补齐训练时的所有 OneHot 列
    for col in encoder_columns:
        if col not in onehot_df.columns:
            onehot_df[col] = 0

    # 只保留训练时的列，并按顺序排列
    onehot_df = onehot_df[encoder_columns]

    # ====== CodeBERT 部分（推理） ======
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

    api_ut_embeds = encode_column(df['api_ut']) if 'api_ut' in df.columns else np.zeros((len(df), codebert_dim))
    tag_embeds = encode_column(df['tags']) if 'tags' in df.columns else np.zeros((len(df), codebert_dim))

    # 拼接所有特征
    X = np.hstack([onehot_df.values, api_ut_embeds, tag_embeds])
    return X

# -----------------------------
# 主推理流程
# -----------------------------
if __name__ == "__main__":
    # 设备
    device = torch.device("cpu")  # 推理用CPU
    print(f"✅ 推理使用设备: {device}")

    # 加载编码器和列顺序
    encoder = joblib.load("encoder.pkl")
    encoder_columns = np.load("encoder_columns.npy", allow_pickle=True)

    # 读取新数据
    csv_path = "new_data.csv"  # TODO: 改成你的推理CSV路径
    df = pd.read_csv(csv_path)

    # 特征处理
    X = process_features_infer(df, encoder, encoder_columns, device)

    # 加载模型
    input_dim = X.shape[1]
    model = LogClassifier(input_dim)
    model.load_state_dict(torch.load("log_classifier.pt", map_location=device))
    model.to(device)
    model.eval()

    # 推理
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # 保存结果
    result_df = df.copy()
    result_df["predicted_label"] = preds
    result_df["prob_class_0"] = probs[:, 0]
    result_df["prob_class_1"] = probs[:, 1]
    result_df.to_csv("inference_results.csv", index=False)

    print("✅ 推理完成，结果已保存到 inference_results.csv")

