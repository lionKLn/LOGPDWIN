import os
import json
import joblib
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Batch

from unsupervised_train.preprocess import generate_graph_in_memory
from model import GAE_GIN
from sklearn.preprocessing import OneHotEncoder


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
# 文本编码工具函数
# ========================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def encode_texts(texts, tokenizer, model, batch_size=32, max_length=128, show_progress=True):
    all_embeddings = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="编码文本")

    with torch.no_grad():
        for start in iterator:
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(outputs, attention_mask).cpu().tolist()
            all_embeddings.extend(sentence_embeddings)

    return all_embeddings


# ========================
# JSON 解析
# ========================
def parse_excel_json(orig_df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for i, row in orig_df.iterrows():
        try:
            data = json.loads(row["data"])
            raw_code = str(data.get("code_str", "")).strip()

            if raw_code.startswith("("):
                processed_code = raw_code
            elif raw_code.startswith("{"):
                processed_code = f"(){raw_code}"
            else:
                processed_code = f"(){{{raw_code}}}"

            results.append({
                "component": data.get("component", ""),
                "case_id": data.get("case_id", ""),
                "test_suite": data.get("test_suite", ""),
                "rule": data.get("rule", ""),
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
                "component": "",
                "case_id": "",
                "test_suite": "",
                "rule": "",
                "code_str": "",
                "raw_code": "",
                "Desc": "",
                "Func": "",
                "case_spce": "",
                "case_purpose": ""
            })

    parsed_df = pd.DataFrame(results)
    merged_df = pd.concat([orig_df.reset_index(drop=True), parsed_df], axis=1)
    return merged_df


# ========================
# 代码图编码
# ========================
def encode_code_graphs(
    merged_df: pd.DataFrame,
    unsupervised_model_path: str,
    embedding_dim: int = 256,
    batch_size: int = 32
) -> pd.DataFrame:
    print("内存中生成 code_str 的代码图...")
    graph_list = []

    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="生成代码图"):
        processed_code = row.get("code_str", "")
        if not processed_code:
            graph_list.append(None)
            continue

        torch_graph = generate_graph_in_memory(
            code_str=processed_code,
            func_name=f"func_{idx}"
        )
        graph_list.append(torch_graph)

    device = get_device()

    print("加载无监督图编码模型...")
    graph_model = GAE_GIN(
        in_channels=768,
        out_channels=768,
        device=device
    )
    graph_model.load_state_dict(torch.load(unsupervised_model_path))
    graph_model.eval()

    print("编码代码图...")
    code_embeddings = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(graph_list), batch_size), desc="编码代码图"):
            batch_graphs = graph_list[batch_start:batch_start + batch_size]
            valid_graphs, valid_indices = [], []

            for idx_in_batch, g in enumerate(batch_graphs):
                if g is not None:
                    valid_graphs.append(g)
                    valid_indices.append(idx_in_batch)

            batch_emb = [torch.zeros(embedding_dim) for _ in batch_graphs]

            if valid_graphs:
                batch = Batch.from_data_list(valid_graphs)
                valid_embs = graph_model.forward(batch, mode="predict")
                for idx_in_batch, emb in zip(valid_indices, valid_embs):
                    batch_emb[idx_in_batch] = emb

            batch_emb_cpu = [emb.cpu().tolist() for emb in batch_emb]
            code_embeddings.extend(batch_emb_cpu)

    merged_df["code_embedding"] = code_embeddings
    return merged_df


# ========================
# 文本字段编码
# ========================
def encode_text_fields(
    merged_df: pd.DataFrame,
    text_model_path: str,
    text_fields=None
) -> pd.DataFrame:
    if text_fields is None:
        text_fields = ["Desc", "Func", "case_spce", "case_purpose"]

    print("加载文本编码模型...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    text_model = AutoModel.from_pretrained(text_model_path)
    text_model.eval()

    for col in text_fields:
        texts = merged_df[col].fillna("").astype(str).tolist()
        merged_df[col + "_embedding"] = encode_texts(texts, tokenizer, text_model)

    return merged_df


# ========================
# One-hot 编码
# ========================
def encode_onehot_features(
    merged_df: pd.DataFrame,
    onehot_encoder_path: str,
    onehot_feature_names_path: str,
    onehot_fields=None,
    mode: str = "infer"   # ===== 修改2：新增 mode 参数 =====
) -> tuple[pd.DataFrame, list]:
    if onehot_fields is None:
        onehot_fields = ["component", "case_id", "test_suite", "rule"]

    if mode not in ["train", "infer"]:  # ===== 修改3：新增 mode 校验 =====
        raise ValueError(f"mode 必须是 'train' 或 'infer'，当前为: {mode}")

    # ===== 修改4：统一先做空值填充与字符串转换 =====
    merged_df[onehot_fields] = merged_df[onehot_fields].fillna("").astype(str)
    onehot_input = merged_df[onehot_fields]

    # ===== 修改5：train 模式，fit_transform 并保存编码器 =====
    if mode == "train":
        print("One-hot 编码：train 模式，拟合并保存编码器...")

        onehot_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

        onehot_features = onehot_encoder.fit_transform(onehot_input)

        # 注意：这里按照你给的训练代码逻辑保存
        joblib.dump(onehot_encoder, onehot_encoder_path)
        np.save(onehot_feature_names_path, onehot_encoder.get_feature_names_out())

        print(f"One-hot编码器已保存至 {onehot_encoder_path} 和 {onehot_feature_names_path}")

        encoder_columns = onehot_encoder.get_feature_names_out().tolist()

        onehot_df = pd.DataFrame(
            onehot_features,
            columns=onehot_encoder.get_feature_names_out()
        )

        merged_df = pd.concat([merged_df, onehot_df], axis=1)
        return merged_df, encoder_columns

    # ===== 修改6：infer 模式，加载已有编码器并 transform =====
    else:
        print("One-hot 编码：infer 模式，加载已有编码器...")

        if not os.path.exists(onehot_encoder_path) or not os.path.exists(onehot_feature_names_path):
            raise FileNotFoundError(
                f"未找到 OneHot 编码器文件，请检查: {onehot_encoder_path}, {onehot_feature_names_path}"
            )

        encoder = joblib.load(onehot_encoder_path)
        encoder_columns = np.load(onehot_feature_names_path, allow_pickle=True).tolist()

        try:
            onehot_encoded = encoder.transform(onehot_input)
        except ValueError as e:
            raise ValueError(f"One-hot 编码失败，输入字段与训练时不一致: {e}")

        # 这里推理时列名要显式传 onehot_fields，保证和 transform 输入一致
        onehot_df = pd.DataFrame(
            onehot_encoded,
            columns=encoder.get_feature_names_out(onehot_fields)
        )

        for col in encoder_columns:
            if col not in onehot_df.columns:
                onehot_df[col] = 0

        onehot_df = onehot_df[encoder_columns]
        merged_df = pd.concat([merged_df, onehot_df], axis=1)

        return merged_df, encoder_columns


# ========================
# 特征融合
# ========================
def merge_features(merged_df: pd.DataFrame, encoder_columns: list) -> pd.DataFrame:
    def _merge_one_row(row):
        code_emb = row["code_embedding"]

        text_embs = []
        for col in ["Desc_embedding", "Func_embedding", "case_spce_embedding", "case_purpose_embedding"]:
            text_embs.extend(row[col])

        onehot_embs = row[encoder_columns].tolist()

        return code_emb + text_embs + onehot_embs

    merged_df["merged_features"] = merged_df.apply(_merge_one_row, axis=1)
    return merged_df


# ========================
# 主流程：Excel -> PKL
# ========================
def encode_excel_to_pkl(
    input_excel: str,
    output_pkl: str,
    unsupervised_model_path: str,
    text_model_path: str,
    onehot_encoder_path: str,
    onehot_feature_names_path: str,
    embedding_dim: int = 256,
    mode: str = "infer"
):
    """
    mode:
        - "infer"：用于推理（无标签）
        - "train"：用于训练（包含标签处理）
    """
    assert mode in ["infer", "train"], f"mode 必须是 infer 或 train，当前是 {mode}"
    set_npu_device()

    if not os.path.exists(input_excel):
        raise FileNotFoundError(f"输入 Excel 不存在: {input_excel}")

    print(f"读取原始 Excel: {input_excel}")
    orig_df = pd.read_excel(input_excel)


    # 1. 解析 JSON
    merged_df = parse_excel_json(orig_df)

    # 标签处理

    if mode == "train":
        print("进入训练模式：处理标签...")

        if "status" not in merged_df.columns:
            raise ValueError("训练模式下必须包含 status 列！")

        status_map = {"t": 1, "f": 0}

        merged_df["false_positive"] = merged_df["status"].map(
            lambda x: status_map.get(str(x).strip().lower(), 0)
        )

        print("标签分布：")
        print(merged_df["false_positive"].value_counts())

    # 2. 代码图编码
    merged_df = encode_code_graphs(
        merged_df=merged_df,
        unsupervised_model_path=unsupervised_model_path,
        embedding_dim=embedding_dim
    )

    # 3. 文本编码
    merged_df = encode_text_fields(
        merged_df=merged_df,
        text_model_path=text_model_path
    )

    # ===== 4：调用 one-hot 时传入 mode =====
    merged_df, encoder_columns = encode_onehot_features(
        merged_df=merged_df,
        onehot_encoder_path=onehot_encoder_path,
        onehot_feature_names_path=onehot_feature_names_path,
        mode=mode
    )

    # 5. 特征融合
    merged_df = merge_features(merged_df, encoder_columns)

    # 6. 保存
    if mode == "train":
        print("保存训练数据（包含标签）...")

        to_save = {
            "orig_df": orig_df,
            "merged_df": merged_df,
            "X": merged_df["merged_features"].tolist(),
            "y": merged_df["false_positive"].tolist()
        }

    else:
        print("保存推理数据（无标签）...")

        to_save = {
            "orig_df": orig_df,
            "merged_df": merged_df
        }

    pd.to_pickle(to_save, output_pkl)
    print(f"待预测数据预处理完成，已保存至: {output_pkl}")


if __name__ == "__main__":
    encode_excel_to_pkl(
        input_excel="../test/temp.xlsx",
        output_pkl="../test/data_to_infer.pkl",
        unsupervised_model_path="../pdg_model/best_pdg.pt",
        text_model_path="../models/paraphrase-multilingual-MiniLM-L12-v2",
        onehot_encoder_path="../onehot_encoder.pkl",
        onehot_feature_names_path="../onehot_feature_names.npy",
        embedding_dim=256,
        mode="infer"
    )
    