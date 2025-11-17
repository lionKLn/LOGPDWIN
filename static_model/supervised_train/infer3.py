import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from model import LogClassifier
import os
import joblib
from sklearn.preprocessing import OneHotEncoder


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
# 配置参数（强调字段顺序必须与训练一致）
# ----------------------------
INPUT_EXCEL = "path/to/your/input.xlsx"
UNSUPERVISED_MODEL_PATH = "logs/pdg/2025-05-20_14-30-00/best_pdg.pt"
DEVICE = torch.device("npu:4" if torch.npu.is_available() else "cpu")
EMBEDDING_DIM = 256
ONEHOT_ENCODER_PATH = "onehot_encoder.pkl"
ONEHOT_FEATURES_PATH = "onehot_feature_names.npy"
# 关键：离散字段顺序必须与训练时fit的顺序完全一致！！！
ONEHOT_FIELDS = ["component", "case_id", "test_suite", "rule"]  # 顺序不可变


# ----------------------------
# 1. 加载Excel与解析JSON
# ----------------------------
df = pd.read_excel(INPUT_EXCEL)

results = []
for i, row in df.iterrows():
    try:
        data = json.loads(row["data"])
        raw_code = str(data.get("code_str", "")).strip()
        if raw_code.startswith('('):
            processed_code = raw_code
        elif raw_code.startswith('{'):
            processed_code = f"(){raw_code}"
        else:
            processed_code = f"(){{{raw_code}}}"

        results.append({
            "component": data.get("component", ""),
            "case_id": data.get("case_id", ""),
            "test_suite": data.get("test_suite", ""),
            "rule": data.get("rule", ""),  # 按ONEHOT_FIELDS顺序排列字段
            "code_str": processed_code,
            "raw_code": raw_code,
            "Desc": data.get("desc", ""),
            "Func": data.get("func", ""),
            "case_spce": data.get("case_spce", ""),
            "case_purpose": data.get("case_purpose", "")
        })
    except Exception as e:
        print(f"第 {i} 行 JSON 解析失败: {e}")
        results.append({
            "component": "", "case_id": "", "test_suite": "", "rule": "",  # 按顺序填充默认值
            "code_str": "", "raw_code": "",
            "Desc": "", "Func": "", "case_spce": "", "case_purpose": ""
        })

merged_df = pd.concat([df, pd.DataFrame(results)], axis=1)


# ----------------------------
# 2. 生成代码图并编码（无修改）
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
# 3. 编码文本字段（无修改）
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
# 4. One-hot 编码（彻底解决字段顺序问题）
# ----------------------------
# 加载编码器和训练时的列名
if not os.path.exists(ONEHOT_ENCODER_PATH) or not os.path.exists(ONEHOT_FEATURES_PATH):
    raise FileNotFoundError(f"❌ 未找到编码器文件，请确保 {ONEHOT_ENCODER_PATH} 和 {ONEHOT_FEATURES_PATH} 存在")

encoder = joblib.load(ONEHOT_ENCODER_PATH)
encoder_columns = np.load(ONEHOT_FEATURES_PATH).tolist()

# 关键：严格按训练时的字段顺序提取数据，确保与encoder.fit时的顺序一致
# 即使merged_df中字段顺序不同，也强制按ONEHOT_FIELDS顺序选取
onehot_input = merged_df[ONEHOT_FIELDS].fillna("").astype(str)

# 执行编码（此时输入字段顺序与训练一致，避免Feature names顺序错误）
try:
    onehot_encoded = encoder.transform(onehot_input)
except ValueError as e:
    raise ValueError(f"❌ 编码失败：输入字段顺序与训练时不一致，请检查ONEHOT_FIELDS是否正确。错误详情：{e}")

# 生成临时DataFrame（列名为编码器自动生成，顺序与训练一致）
onehot_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out(ONEHOT_FIELDS))

# 补齐训练时的列（若有缺失）并强制按训练顺序排列
for col in encoder_columns:
    if col not in onehot_df.columns:
        onehot_df[col] = 0
onehot_df = onehot_df[encoder_columns]  # 最终顺序与训练完全一致

# 拼接至合并数据
merged_df = pd.concat([merged_df, onehot_df], axis=1)


# ----------------------------
# 5. 特征融合（使用训练时的列名顺序）
# ----------------------------
def merge_features(row):
    code_emb = row["code_embedding"]
    text_embs = []
    for col in ["Desc_embedding", "Func_embedding", "case_spce_embedding", "case_purpose_embedding"]:
        text_embs.extend(row[col])
    onehot_embs = row[encoder_columns].tolist()  # 按训练列名顺序提取
    return code_emb + text_embs + onehot_embs

merged_df["merged_features"] = merged_df.apply(merge_features, axis=1)


# ----------------------------
# 6. 保存结果
# ----------------------------
processed_data_path = "data_to_infer.pkl"
merged_df.to_pickle(processed_data_path)
print(f"✅ 待分析数据已处理完成，保存至 {processed_data_path}")


# ========================
# 🔧 推理配置与函数（无修改）
# ========================
MODEL_PATH = "best_log_classifier.pt"
DATA_PATH = "data_to_infer.pkl"
DEVICE = torch.device("npu:5" if torch.npu.is_available() else "cpu")
HIDDEN_DIM = 128
OUTPUT_PATH = "inference_results.csv"

def load_new_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 数据文件不存在: {data_path}")
    data = pd.read_pickle(data_path)
    X_new = torch.tensor(data["merged_features"].tolist(), dtype=torch.float32)
    print(f"✅ 已加载新数据，共 {len(X_new)} 条样本。")
    return data, X_new

def predict_with_prob(model_path, data_tensor, hidden_dim=128):
    input_dim = data_tensor.shape[1]
    model = LogClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(data_tensor.to(DEVICE))
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs[:, 0], probs[:, 1]

if __name__ == "__main__":
    print("🚀 开始模型推理...")
    original_df, X_new = load_new_data(DATA_PATH)
    preds, prob_0, prob_1 = predict_with_prob(MODEL_PATH, X_new, hidden_dim=HIDDEN_DIM)
    print("✅ 推理完成！")
    result_df = original_df.copy()
    result_df["pred_label"] = preds
    result_df["prob_0"] = prob_0
    result_df["prob_1"] = prob_1
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"📄 预测结果已保存至：{OUTPUT_PATH}")
    print(f"样例预览：")
    print(result_df.head())