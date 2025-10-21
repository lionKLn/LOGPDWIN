import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Batch  # 新增：用于合并内存中的图数据
# 导入内存版代码图生成函数（替换原process_sample）
from unsupervised_train.preprocess import generate_graph_in_memory  # 关键替换：内存版函数
from model import GAE_GIN  # 无监督训练的图编码模型

# ----------------------------
# 配置参数（删除GRAPH_SAVE_DIR，无需临时目录）
# ----------------------------
INPUT_EXCEL = "path/to/your/input.xlsx"  # 原始Excel路径
UNSUPERVISED_MODEL_PATH = "logs/pdg/2025-05-20_14-30-00/best_pdg.pt"  # 无监督模型路径
DEVICE = torch.device("npu:4" if torch.npu.is_available() else "cpu")  # 设备
EMBEDDING_DIM = 256  # 无监督模型输出的嵌入维度（需与模型一致）

# ----------------------------
# 1. 加载数据与预处理（解析JSON）
# ----------------------------
# 读取原始Excel
df = pd.read_excel(INPUT_EXCEL)

# 提取JSON字段并处理code_str（逻辑不变）
results = []
for i, row in df.iterrows():
    try:
        data = json.loads(row["data"])
        # 原始code_str提取
        raw_code = str(data.get("code_str", "")).strip()

        # ---------------------- code_str预处理（复用之前的规则） ----------------------
        if raw_code.startswith('('):
            processed_code = raw_code  # 以"("开头：不处理
        elif raw_code.startswith('{'):
            processed_code = f"(){raw_code}"  # 以"{"开头：最前面加"()"
        else:
            processed_code = f"(){{{raw_code}}}"  # 其他开头：先套"{}"再在最前加"()"
        # --------------------------------------------------------------------------

        results.append({
            "component": data.get("component", ""),
            "code_str": processed_code,  # 存储预处理后的code_str
            "raw_code": raw_code,  # 保留原始code_str用于追溯
            "Desc": data.get("desc", ""),
            "Func": data.get("func", ""),
            "case_id": data.get("case_id", ""),
            "test_suite": data.get("test_suite", ""),
            "case_spce": data.get("case_spce", ""),
            "case_purpose": data.get("case_purpose", "")
        })
    except Exception as e:
        print(f"第 {i} 行JSON解析失败: {e}")
        results.append({
            "component": "", "code_str": "", "raw_code": "",
            "Desc": "", "Func": "", "case_id": "", "test_suite": "",
            "case_spce": "", "case_purpose": ""
        })

# 合并原始数据与解析结果
new_df = pd.DataFrame(results)
merged_df = pd.concat([df, new_df], axis=1)

# ----------------------------
# 2. code_str处理：内存生成代码图→直接编码（核心修改）
# ----------------------------
# 2.1 内存批量生成代码图（不保存文件）
print("内存中生成code_str的代码图...")
graph_list = []  # 存储所有内存中的PyG Data对象（与merged_df行对齐）
for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="生成代码图"):
    processed_code = row["code_str"]
    if not processed_code:  # 跳过空code_str
        graph_list.append(None)
        continue

    # 调用内存版函数，直接返回torch_graph（PyG Data对象）
    # func_name用idx确保唯一，避免解析冲突
    torch_graph = generate_graph_in_memory(
        code_str=processed_code,
        func_name=f"func_{idx}"
    )
    graph_list.append(torch_graph)  # 存入列表，后续直接使用

# 2.2 加载无监督模型（逻辑不变）
print("加载无监督图编码模型...")
graph_model = GAE_GIN(
    in_channels=768,  # 节点特征维度（CodeBERT编码维度）
    out_channels=768,
    device=DEVICE
).to(DEVICE)
graph_model.load_state_dict(torch.load(UNSUPERVISED_MODEL_PATH, map_location=DEVICE))
graph_model.eval()  # 切换到评估模式

# 2.3 内存批量编码代码图（无文件读取）
print("编码内存中的代码图...")
code_embeddings = []
batch_size = 32  # 按批次处理，避免内存溢出

with torch.no_grad():  # 禁用梯度计算，节省内存
    # 按批次遍历graph_list
    for batch_start in tqdm(range(0, len(graph_list), batch_size), desc="编码代码图"):
        # 取当前批次的图列表
        batch_graphs = graph_list[batch_start:batch_start + batch_size]

        # 1. 分离有效图（torch_graph非None）和无效图（None）
        valid_graphs = []  # 存储有效PyG Data对象
        valid_indices = []  # 有效图在当前批次的索引
        for idx_in_batch, g in enumerate(batch_graphs):
            if g is not None:
                valid_graphs.append(g)
                valid_indices.append(idx_in_batch)

        # 2. 对有效图进行批量编码
        batch_emb = [torch.zeros(EMBEDDING_DIM, device=DEVICE) for _ in batch_graphs]  # 初始化批次嵌入
        if valid_graphs:
            # 合并有效图为PyG Batch对象（内存中直接合并）
            batch = Batch.from_data_list(valid_graphs).to(DEVICE)
            # 调用模型编码，得到图嵌入
            valid_embs = graph_model.forward(batch, mode="predict")  # 形状：[有效图数量, EMBEDDING_DIM]

            # 将有效嵌入赋值到对应位置
            for idx_in_batch, emb in zip(valid_indices, valid_embs):
                batch_emb[idx_in_batch] = emb

        # 3. 将当前批次嵌入移到CPU并转为列表（便于后续存入DataFrame）
        batch_emb_cpu = [emb.cpu().tolist() for emb in batch_emb]
        code_embeddings.extend(batch_emb_cpu)

# 将编码结果存入merged_df（与行严格对齐）
merged_df["code_embedding"] = code_embeddings

# ----------------------------
# 3. 其他字段编码（复用原有逻辑，无修改）
# ----------------------------
# Sentence-BERT编码文本字段
MODEL_PATH = "./models/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text_model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
text_model.eval()

def mean_pooling(model_output, attention_mask):
    """
    对 transformer 的 last_hidden_state 按 attention_mask 做加权平均作为句向量
    model_output: transformers 输出对象（含 last_hidden_state）
    attention_mask: tensor shape (batch, seq_len)
    返回: tensor shape (batch, hidden_size)
    """
    token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()  # (batch, seq_len, hidden)
    # 避免除以0
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)  # (batch, hidden)
    # 防止除0（当某行全是padding时）
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask

def encode_texts(texts, tokenizer, model, device, batch_size=32, max_length=128, show_progress=True):
    """
    使用 transformers.AutoTokenizer + AutoModel 编码文本为向量（mean pooling）。
    texts: list[str]
    tokenizer: AutoTokenizer 实例
    model: AutoModel 实例（已 .to(device) 并 eval()）
    device: torch.device（例如 'cpu','cuda' 或 'npu'）
    batch_size: 批大小
    max_length: 最大序列长度
    返回: list of list (每个文本对应的向量)
    """
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="编码文本")
        for start in iterator:
            batch_texts = texts[start:start + batch_size]
            # tokenizer 返回 tensors 放在 device 上
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
            # mean pooling（按mask）
            sentence_embeddings = mean_pooling(outputs, attention_mask)  # (batch, hidden)
            # 转为 CPU list
            sentence_embeddings = sentence_embeddings.cpu().tolist()
            all_embeddings.extend(sentence_embeddings)

    return all_embeddings

# def encode_texts(texts):
#     return text_model.encode([str(x) if x is not None else "" for x in texts], show_progress_bar=True)


for col in ["Desc", "Func", "case_spce", "case_purpose"]:
    texts = merged_df[col].fillna("").astype(str).tolist()
    embeddings = encode_texts(texts, tokenizer, text_model, DEVICE, batch_size=32, max_length=128)
    merged_df[col + "_embedding"] = [emb for emb in embeddings]

# One-hot编码类别字段
component_onehot = pd.get_dummies(merged_df["component"], prefix="component")
case_id_onehot = pd.get_dummies(merged_df["case_id"], prefix="case_id")
test_suite_onehot = pd.get_dummies(merged_df["test_suite"], prefix="test_suite")
rule_onehot = pd.get_dummies(merged_df["rule"], prefix="rule")

# ----------------------------
# 4. 标签处理（复用原有逻辑，无修改）
# ----------------------------
status_map = {"t": 1, "f": 0}
merged_df["false_positive"] = merged_df["status"].map(
    lambda x: status_map.get(str(x).strip().lower(), 0)
)
print("标签分布：")
print(merged_df["false_positive"].value_counts())


# ----------------------------
# 5. 特征融合（复用原有逻辑，无修改）
# ----------------------------
def merge_features(row):
    """将所有特征拼接为一个向量"""
    # 1. 代码图嵌入（code_embedding）
    code_emb = row["code_embedding"]

    # 2. 文本字段嵌入（Desc_embedding等）
    text_embs = []
    for col in ["Desc_embedding", "Func_embedding", "case_space_embedding", "case_purpose_embedding"]:
        text_embs.extend(row[col])

    # 3. One-hot特征（需先将One-hot编码表与merged_df合并）
    onehot_cols = [c for c in row.index if c.startswith(("component_", "case_id_", "test_suite_", "rule_"))]
    onehot_embs = row[onehot_cols].tolist()

    # 拼接所有特征
    return code_emb + text_embs + onehot_embs


# 合并One-hot编码到merged_df
merged_df = pd.concat([merged_df, component_onehot, case_id_onehot, test_suite_onehot, rule_onehot], axis=1)

# 生成最终融合特征
merged_df["merged_features"] = merged_df.apply(merge_features, axis=1)

# 保存处理后的数据集（用于后续有监督训练）
processed_data_path = "processed_dataset.pkl"
merged_df.to_pickle("processed_data_path")
print(f"所有特征处理完成，已保存至 {processed_data_path}")

# 新增：训练相关配置
TRAINED_MODEL_SAVE_PATH = "best_log_classifier.pt"  # 训练好的模型保存路径
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-4
HIDDEN_DIM = 128
TEST_SIZE = 0.2
RANDOM_SEED = 42
POS_LABEL = 0  # 核心关注0类（非误报）
NEG_LABEL = 1# ----------------------------
# 6. 调用训练流程（特征保存后新增）
# ----------------------------
print("\n===== 开始调用模型训练流程 =====")
# 导入训练模块（确保train_script.py与当前文件在同一目录，或已加入环境变量）
from train_supervised import train_model  # 假设之前封装的训练代码在train_script.py中

# 调用训练函数，使用生成的特征文件作为输入
try:
    # 训练模型并获取结果
    trained_model, final_metrics = train_model(
        data_path=processed_data_path,  # 使用刚生成的特征文件
        save_model_path=TRAINED_MODEL_SAVE_PATH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
        pos_label=POS_LABEL  # 核心关注0类
    )
    print("\n===== 训练流程完成 =====")
    print(f"最优模型已保存至：{TRAINED_MODEL_SAVE_PATH}")
    print("核心指标 summary：")
    print(f"0类F1分数：{final_metrics['class_0']['f1']:.4f}")
    print(f"0类召回率：{final_metrics['class_0']['recall']:.4f}")
except Exception as e:
    print(f"训练过程出错：{e}")