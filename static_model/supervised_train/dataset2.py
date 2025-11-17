import os
import json
import pandas as pd
import torch
import joblib  # 新增：用于保存编码器
import numpy as np  # 新增：用于保存特征列名
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Batch
from unsupervised_train.preprocess import generate_graph_in_memory
from model import GAE_GIN
from sklearn.preprocessing import OneHotEncoder  # 新增：替换pd.get_dummies

# ----------------------------
# 配置参数（新增编码器保存路径）
# ----------------------------
INPUT_EXCEL = "path/to/your/input.xlsx"
UNSUPERVISED_MODEL_PATH = "logs/pdg/2025-05-20_14-30-00/best_pdg.pt"
DEVICE = torch.device("npu:4" if torch.npu.is_available() else "cpu")
EMBEDDING_DIM = 256
ONEHOT_ENCODER_PATH = "onehot_encoder.pkl"  # 新增：编码器保存路径
ONEHOT_FEATURES_PATH = "onehot_feature_names.npy"  # 新增：特征列名保存路径

# ----------------------------
# 1. 加载数据与预处理（解析JSON）
# ----------------------------
df = pd.read_excel(INPUT_EXCEL)

results = []
for i, row in df.iterrows():
    try:
        data = json.loads(row["data"])
        raw_code = str(data.get("code_str", "")).strip()

        # code_str预处理
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
            "case_purpose": data.get("case_purpose", ""),
            "rule": data.get("rule", "")  # 确保rule字段被提取
        })
    except Exception as e:
        print(f"第 {i} 行JSON解析失败: {e}")
        results.append({
            "component": "", "code_str": "", "raw_code": "",
            "Desc": "", "Func": "", "case_id": "", "test_suite": "",
            "case_spce": "", "case_purpose": "", "rule": ""
        })

new_df = pd.DataFrame(results)
merged_df = pd.concat([df, new_df], axis=1)

# ----------------------------
# 2. 代码图生成与编码（无修改）
# ----------------------------
print("内存中生成code_str的代码图...")
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

print("编码内存中的代码图...")
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
# 3. 文本字段编码（无修改）
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
    model.eval()
    with torch.no_grad():
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="编码文本")
        for start in iterator:
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(outputs, attention_mask).cpu().tolist()
            all_embeddings.extend(sentence_embeddings)
    return all_embeddings


for col in ["Desc", "Func", "case_spce", "case_purpose"]:
    texts = merged_df[col].fillna("").astype(str).tolist()
    embeddings = encode_texts(texts, tokenizer, text_model, DEVICE, batch_size=32, max_length=128)
    merged_df[col + "_embedding"] = embeddings

# ----------------------------
# 4. One-hot编码（核心修改：使用OneHotEncoder并保存）
# ----------------------------
# 定义需要编码的离散字段
categorical_cols = ["component", "case_id", "test_suite", "rule"]

# 填充空值并转为字符串（避免编码器报错）
merged_df[categorical_cols] = merged_df[categorical_cols].fillna("").astype(str)

# 初始化编码器（handle_unknown='ignore'处理未知特征）
onehot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"  # 关键：遇到训练未见过的特征时输出全0
)

# 拟合数据并生成One-hot特征
onehot_features = onehot_encoder.fit_transform(merged_df[categorical_cols])

# 保存编码器和特征列名（供推理时使用）
joblib.dump(onehot_encoder, ONEHOT_ENCODER_PATH)
np.save(ONEHOT_FEATURES_PATH, onehot_encoder.get_feature_names_out())
print(f"One-hot编码器已保存至 {ONEHOT_ENCODER_PATH} 和 {ONEHOT_FEATURES_PATH}")

# 将编码结果转为DataFrame并拼接
onehot_df = pd.DataFrame(
    onehot_features,
    columns=onehot_encoder.get_feature_names_out()
)
merged_df = pd.concat([merged_df, onehot_df], axis=1)

# ----------------------------
# 5. 标签处理（无修改）
# ----------------------------
status_map = {"t": 1, "f": 0}
merged_df["false_positive"] = merged_df["status"].map(
    lambda x: status_map.get(str(x).strip().lower(), 0)
)
print("标签分布：")
print(merged_df["false_positive"].value_counts())


# ----------------------------
# 6. 特征融合（修改：适配OneHotEncoder的列名）
# ----------------------------
def merge_features(row):
    code_emb = row["code_embedding"]

    text_embs = []
    # 注意：原代码中"case_space_embedding"应为笔误，修正为"case_spce_embedding"
    for col in ["Desc_embedding", "Func_embedding", "case_spce_embedding", "case_purpose_embedding"]:
        text_embs.extend(row[col])

    # 从OneHotEncoder生成的列名中提取特征
    onehot_cols = onehot_encoder.get_feature_names_out().tolist()
    onehot_embs = row[onehot_cols].tolist()

    return code_emb + text_embs + onehot_embs


merged_df["merged_features"] = merged_df.apply(merge_features, axis=1)

# ----------------------------
# 7. 保存处理后的数据（无修改）
# ----------------------------
processed_data_path = "processed_dataset.pkl"
merged_df.to_pickle(processed_data_path)
print(f"所有特征处理完成，已保存至 {processed_data_path}")

# ----------------------------
# 8. 训练配置与调用（无修改）
# ----------------------------
TRAINED_MODEL_SAVE_PATH = "best_log_classifier.pt"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-4
HIDDEN_DIM = 128
TEST_SIZE = 0.2
RANDOM_SEED = 42
POS_LABEL = 0
NEG_LABEL = 1

print("\n===== 开始调用模型训练流程 =====")
from train_supervised import train_model

try:
    trained_model, final_metrics = train_model(
        data_path=processed_data_path,
        save_model_path=TRAINED_MODEL_SAVE_PATH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
        pos_label=POS_LABEL
    )
    print("\n===== 训练流程完成 =====")
    print(f"最优模型已保存至：{TRAINED_MODEL_SAVE_PATH}")
    print("核心指标 summary：")
    print(f"0类F1分数：{final_metrics['class_0']['f1']:.4f}")
    print(f"0类召回率：{final_metrics['class_0']['recall']:.4f}")
except Exception as e:
    print(f"训练过程出错：{e}")