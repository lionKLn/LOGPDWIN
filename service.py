import pandas as pd
import torch
import numpy as np
import joblib
from fastapi import FastAPI, Body
from infer import process_features_infer, LogClassifier

app = FastAPI(title="Huawei Log Inference Service (JSON output)")

# -----------------------------
# 设备 & 模型预加载
# -----------------------------
device = torch.device("cpu")

encoder = joblib.load("encoder.pkl")
encoder_columns = np.load("encoder_columns.npy", allow_pickle=True)

# 初始化模型（先占位一个输入维度）
dummy_input = np.zeros((1, len(encoder_columns) + 768*2))  # onehot + api_ut + tags
input_dim = dummy_input.shape[1]
model = LogClassifier(input_dim)
model.load_state_dict(torch.load("log_classifier.pt", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 接口定义：接收 JSON -> 输出 JSON
# -----------------------------
@app.post("/api/predict")
async def infer_json(payload: list = Body(...)):
    """
    输入：JSON 数组
    [
      {
        "pipeline_id": "",
        "case_id": "",
        "test_data_id": "",
        "api_ui": "",
        "oracle_name": "",
        "tags": ["ais","oracleAsian","ais_zone"],
        "error_id": "",
        "error_id_old": "",
        "component": "",
        "module": ""
      }
    ]
    输出：每条数据增加预测结果字段
    """
    # 转换 JSON 为 DataFrame
    df = pd.DataFrame(payload)

    # 字段映射：和训练时保持一致
    df = df.rename(columns={
        "api_ui": "api_ut",
        "component": "sut.component",
        "module": "sut.module"
    })

    # 处理 tags（拼接成字符串供编码）
    df["tags"] = df["tags"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

    # 补齐缺失字段
    if "sut.component_set" not in df.columns:
        df["sut.component_set"] = ""

    # 特征处理
    X = process_features_infer(df, encoder, encoder_columns, device)

    # 推理
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # 添加预测结果到每条记录
    result_list = []
    for i, row in df.iterrows():
        record = row.to_dict()
        record["predicted_label"] = int(preds[i])
        record["prob_class_0"] = float(probs[i, 0])
        record["prob_class_1"] = float(probs[i, 1])
        result_list.append(record)

    return result_list
