import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import LogClassifier


# ========================
# 设备设置
# ========================
def get_device():
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device("npu:5")
    except Exception:
        pass
    return torch.device("cpu")


def set_npu_device():
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.set_device(5)
            print("当前已显式设置 NPU 设备为 npu:5")
        else:
            print("当前环境未检测到可用 NPU，使用 CPU")
    except Exception as e:
        print(f"设置 NPU 设备失败，回退到 CPU。原因: {e}")


# ========================
# 加载待预测数据
# ========================
def load_processed_pkl(data_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    loaded = pd.read_pickle(data_path)

    if isinstance(loaded, dict):
        orig_df = loaded.get("orig_df")
        merged_df = loaded.get("merged_df")
    else:
        orig_df = None
        merged_df = loaded

    if merged_df is None:
        raise ValueError("未从 pkl 中找到 merged_df")

    X_new = torch.tensor(merged_df["merged_features"].tolist(), dtype=torch.float32)

    print(f"已加载待预测数据，共 {len(X_new)} 条样本")
    return orig_df, merged_df, X_new


# ========================
# 预测
# ========================
def predict_with_prob(
    model_path: str,
    data_tensor: torch.Tensor,
    hidden_dim: int = 128
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    input_dim = data_tensor.shape[1]

    model = LogClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        outputs = model(data_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    return preds, probs[:, 0], probs[:, 1]


# ========================
# 保存结果
# ========================
def save_prediction_results(
    orig_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    preds,
    prob_0,
    prob_1,
    output_csv: str
):
    n_samples = len(preds)

    if orig_df is None:
        print("警告：未找到原始 orig_df，将使用 merged_df 作为基础结果表")
        base_df = merged_df.copy().reset_index(drop=True)
    else:
        base_df = orig_df.copy().reset_index(drop=True)

    if len(base_df) != n_samples:
        min_len = min(len(base_df), n_samples)
        print(f"数据条目不一致，按最小长度 {min_len} 截断")
        base_df = base_df.iloc[:min_len].reset_index(drop=True)
        preds = preds[:min_len]
        prob_0 = prob_0[:min_len]
        prob_1 = prob_1[:min_len]

    result_df = base_df.copy()
    result_df["pred_label"] = preds
    result_df["prob_0"] = prob_0
    result_df["prob_1"] = prob_1

    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"预测结果已保存至: {output_csv}")
    print("结果预览：")
    print(result_df.head())


# ========================
# 主流程：PKL -> CSV
# ========================
def predict_pkl_to_csv(
    input_pkl: str,
    model_path: str,
    output_csv: str,
    hidden_dim: int = 128
):
    set_npu_device()

    print("开始模型推理...")
    orig_df, merged_df, X_new = load_processed_pkl(input_pkl)

    preds, prob_0, prob_1 = predict_with_prob(
        model_path=model_path,
        data_tensor=X_new,
        hidden_dim=hidden_dim
    )

    print("推理完成，开始保存结果...")
    save_prediction_results(
        orig_df=orig_df,
        merged_df=merged_df,
        preds=preds,
        prob_0=prob_0,
        prob_1=prob_1,
        output_csv=output_csv
    )


if __name__ == "__main__":
    predict_pkl_to_csv(
        input_pkl="data_to_infer.pkl",
        model_path="best_log_classifier.pt",
        output_csv="inference_results.csv",
        hidden_dim=128
    )