import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from model import LogClassifier
import os




import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Batch
from unsupervised_train.preprocess import generate_graph_in_memory
from model import GAE_GIN

# ----------------------------
# 配置参数
# ----------------------------
INPUT_EXCEL = "path/to/your/input.xlsx"  # 待分析数据
UNSUPERVISED_MODEL_PATH = "logs/pdg/2025-05-20_14-30-00/best_pdg.pt"
DEVICE = torch.device("npu:4" if torch.npu.is_available() else "cpu")
EMBEDDING_DIM = 256

# ----------------------------
# 1. 加载Excel与解析JSON
# ----------------------------
df = pd.read_excel(INPUT_EXCEL)

results = []
for i, row in df.iterrows():
    try:
        data = json.loads(row["data"])
        raw_code = str(data.get("code_str", "")).strip()
        # code_str 预处理
        if raw_code.startswith('('):
            processed_code = raw_code
        elif raw_code.startswith('{'):
            processed_code = f"(){raw_code}"
        else:
            processed_code = f"(){{{raw_code}}}"

        results.append({
            "component": data.get("component", ""),
            "code_str": processed_code,
            "raw_code": raw_code,
            "Desc": data.get("desc", ""),
            "Func": data.get("func", ""),
            "case_id": data.get("case_id", ""),
            "test_suite": data.get("test_suite", ""),
            "case_spce": data.get("case_spce", ""),
            "case_purpose": data.get("case_purpose", "")
        })
    except Exception as e:
        print(f"第 {i} 行 JSON 解析失败: {e}")
        results.append({
            "component": "", "code_str": "", "raw_code": "",
            "Desc": "", "Func": "", "case_id": "", "test_suite": "",
            "case_spce": "", "case_purpose": ""
        })

merged_df = pd.concat([df, pd.DataFrame(results)], axis=1)

# ----------------------------
# 2. 生成代码图并编码（无监督模型）
# ----------------------------
print("内存中生成 code_str 的代码图...")
graph_list = []
for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="生成代码图"):
    processed_code = row["code_str"]
    if not processed_code:
        graph_list.append(None)
        continue

    torch_graph = generate_graph_in_memory(
        code_str=processed_code,
        func_name=f"func_{idx}"
    )
    graph_list.append(torch_graph)

print("加载无监督图编码模型...")
graph_model = GAE_GIN(
    in_channels=768,
    out_channels=768,
    device=DEVICE
).to(DEVICE)
graph_model.load_state_dict(torch.load(UNSUPERVISED_MODEL_PATH, map_location=DEVICE))
graph_model.eval()

print("编码代码图...")
code_embeddings = []
batch_size = 32

with torch.no_grad():
    for batch_start in tqdm(range(0, len(graph_list), batch_size), desc="编码代码图"):
        batch_graphs = graph_list[batch_start:batch_start + batch_size]
        valid_graphs, valid_indices = [], []
        for idx_in_batch, g in enumerate(batch_graphs):
            if g is not None:
                valid_graphs.append(g)
                valid_indices.append(idx_in_batch)

        batch_emb = [torch.zeros(EMBEDDING_DIM, device=DEVICE) for _ in batch_graphs]
        if valid_graphs:
            batch = Batch.from_data_list(valid_graphs).to(DEVICE)
            valid_embs = graph_model.forward(batch, mode="predict")
            for idx_in_batch, emb in zip(valid_indices, valid_embs):
                batch_emb[idx_in_batch] = emb

        batch_emb_cpu = [emb.cpu().tolist() for emb in batch_emb]
        code_embeddings.extend(batch_emb_cpu)

merged_df["code_embedding"] = code_embeddings

# ----------------------------
# 3. 编码文本字段
# ----------------------------
MODEL_PATH = "./models/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text_model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
text_model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def encode_texts(texts, tokenizer, model, device, batch_size=32, max_length=128, show_progress=True):
    all_embeddings = []
    with torch.no_grad():
        iterator = tqdm(range(0, len(texts), batch_size), desc="编码文本") if show_progress else range(0, len(texts), batch_size)
        for start in iterator:
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids, attention_mask = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(outputs, attention_mask).cpu().tolist()
            all_embeddings.extend(sentence_embeddings)
    return all_embeddings

for col in ["Desc", "Func", "case_spce", "case_purpose"]:
    texts = merged_df[col].fillna("").astype(str).tolist()
    merged_df[col + "_embedding"] = encode_texts(texts, tokenizer, text_model, DEVICE)

# ----------------------------
# 4. One-hot 编码
# ----------------------------
component_onehot = pd.get_dummies(merged_df["component"], prefix="component")
case_id_onehot = pd.get_dummies(merged_df["case_id"], prefix="case_id")
test_suite_onehot = pd.get_dummies(merged_df["test_suite"], prefix="test_suite")
rule_onehot = pd.get_dummies(merged_df.get("rule", pd.Series([])), prefix="rule")

merged_df = pd.concat([merged_df, component_onehot, case_id_onehot, test_suite_onehot, rule_onehot], axis=1)

# ----------------------------
# 5. 特征融合（无标签）
# ----------------------------
def merge_features(row):
    code_emb = row["code_embedding"]
    text_embs = []
    for col in ["Desc_embedding", "Func_embedding", "case_spce_embedding", "case_purpose_embedding"]:
        text_embs.extend(row[col])
    onehot_cols = [c for c in row.index if c.startswith(("component_", "case_id_", "test_suite_", "rule_"))]
    onehot_embs = row[onehot_cols].tolist()
    return code_emb + text_embs + onehot_embs

merged_df["merged_features"] = merged_df.apply(merge_features, axis=1)

# ----------------------------
# 6. 保存结果（无标签数据）
# ----------------------------
processed_data_path = "data_to_infer.pkl"
merged_df.to_pickle(processed_data_path)
print(f"✅ 待分析数据已处理完成，保存至 {processed_data_path}")


# ========================
# 🔧 推理配置
# ========================
MODEL_PATH = "best_log_classifier.pt"   # 模型路径
DATA_PATH = "data_to_infer.pkl"              # 已编码的新数据路径
DEVICE = torch.device("npu:5" if torch.npu.is_available() else "cpu")
HIDDEN_DIM = 128                        # 与训练时一致
OUTPUT_PATH = "inference_results.csv"   # 输出结果文件路径


# ========================
# 🔹 数据加载函数
# ========================
def load_new_data(data_path):
    """
    加载新数据并转换为tensor格式
    假设DataFrame中含有一列 'merged_features'（与训练阶段一致）
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 数据文件不存在: {data_path}")

    data = pd.read_pickle(data_path)
    X_new = torch.tensor(data["merged_features"].tolist(), dtype=torch.float32)
    print(f"✅ 已加载新数据，共 {len(X_new)} 条样本。")
    return data, X_new


# ========================
# 🔹 推理函数
# ========================
def predict_with_prob(model_path, data_tensor, hidden_dim=128):
    """
    使用保存的模型对新数据进行预测，并输出概率
    :param model_path: 模型文件路径
    :param data_tensor: 新样本特征张量 (shape: [N, feature_dim])
    :param hidden_dim: 模型隐藏层维度（与训练保持一致）
    :return: (pred_labels, probs_0, probs_1)
    """
    # 1️⃣ 加载模型结构
    input_dim = data_tensor.shape[1]
    model = LogClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2️⃣ 推理阶段
    with torch.no_grad():
        outputs = model(data_tensor.to(DEVICE))              # [N, 2]
        probs = F.softmax(outputs, dim=1).cpu().numpy()      # 转换为概率分布
        preds = np.argmax(probs, axis=1)                     # 取最大概率对应的标签

    return preds, probs[:, 0], probs[:, 1]


# ========================
# 🔹 主程序入口
# ========================
if __name__ == "__main__":
    print("🚀 开始模型推理...")

    # 加载数据
    original_df, X_new = load_new_data(DATA_PATH)

    # 模型预测
    preds, prob_0, prob_1 = predict_with_prob(MODEL_PATH, X_new, hidden_dim=HIDDEN_DIM)
    print("✅ 推理完成！")

    # 组合结果
    result_df = original_df.copy()
    result_df["pred_label"] = preds
    result_df["prob_0"] = prob_0
    result_df["prob_1"] = prob_1

    # 输出保存
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"📄 预测结果已保存至：{OUTPUT_PATH}")
    print(f"样例预览：")
    print(result_df.head())
