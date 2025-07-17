import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import argparse
from model import LogClassifier  # 确保你的模型结构保存在 model.py 中并可导入
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to inference data CSV file')
    parser.add_argument('--model', type=str, default='log_classifier.pt', help='Path to model file')
    parser.add_argument('--encoder', type=str, default='encoder.pkl', help='Path to encoder pkl file')
    parser.add_argument('--columns', type=str, default='encoder_columns.npy', help='Path to encoder columns file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to save prediction results')
    args = parser.parse_args()

    device = torch.device("cpu")

    # 加载模型结构并加载权重
    model = LogClassifier(input_dim=None)  # 暂时为 None
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 加载编码器和编码列顺序
    encoder = joblib.load(args.encoder)
    encoder_columns = np.load(args.columns, allow_pickle=True)

    # 读取推理数据
    df = pd.read_csv(args.data)

    # 特征处理：保持与训练时相同的列顺序
    X_encoded = encoder.transform(df)

    # 将稀疏矩阵转为稠密矩阵（如果适用）
    if hasattr(X_encoded, "toarray"):
        X_encoded = X_encoded.toarray()

    # 转换为 DataFrame 以补齐缺失列
    X_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out())

    # 填充缺失列
    for col in encoder_columns:
        if col not in X_df.columns:
            X_df[col] = 0
    X_df = X_df[encoder_columns]  # 保证顺序一致

    # 转换为 tensor
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)

    # 推理
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    # 保存结果
    result_df = df.copy()
    result_df['predicted_label'] = preds
    result_df['prob_class_0'] = probs[:, 0]
    result_df['prob_class_1'] = probs[:, 1]

    result_df.to_csv(args.output, index=False)
    print(f"✅ 预测结果已保存至: {args.output}")

if __name__ == "__main__":
    main()
