import pandas as pd
import torch
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from infer import process_features_infer, LogClassifier
import tempfile
import os

app = FastAPI(title="Huawei Log Inference Service")

# -----------------------------
# 设备 & 模型预加载
# -----------------------------
device = torch.device("cpu")

encoder = joblib.load("encoder.pkl")
encoder_columns = np.load("encoder_columns.npy", allow_pickle=True)

# 初始化模型
dummy_input = np.zeros((1, len(encoder_columns) + 768*2))  # onehot + api_ut + tags
input_dim = dummy_input.shape[1]
model = LogClassifier(input_dim)
model.load_state_dict(torch.load("log_classifier.pt", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 接口定义
# -----------------------------
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # 读取上传的 CSV
    df = pd.read_csv(file.file)

    # 特征处理
    X = process_features_infer(df, encoder, encoder_columns, device)

    # 推理
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # 保存结果到临时文件
    result_df = df.copy()
    result_df["predicted_label"] = preds
    result_df["prob_class_0"] = probs[:, 0]
    result_df["prob_class_1"] = probs[:, 1]

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    result_df.to_csv(tmp_file.name, index=False)

    return FileResponse(
        tmp_file.name,
        media_type="text/csv",
        filename="inference_results.csv"
    )
