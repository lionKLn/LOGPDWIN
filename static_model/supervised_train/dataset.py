import pandas as pd
import numpy as np
import torch
import json

os
import io
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaModel

# 导入preprocess相关模块（你的CPG生成代码）
from ccpg.sast.src_parser import c_parser_from_serial_code
from ccpg.sast.fun_unit import FunUnit
from ccpg.cpg.ast_constructor import gen_ast_cpg
from ccpg.cpg.cfg_constructor import cfg_build
from ccpg.cpg.ddg_constructor import ddg_build
from ccpg.cpg.cpg_node import CPGNode

# 导入模型模块
from unsupervised_train.model import GAE_GIN_lightning

# ----------------------------
# 配置与初始化
# ----------------------------
# 设备配置
device = torch.device("npu:6" if torch.npu.is_available() else
                      "cuda:0" if torch.cuda.is_available() else "cpu")

# 模型路径（训练好的InfoGraph模型）
MODEL_CHECKPOINT = "./checkpoints/best/gin-unsupervised-best.ckpt"

# 输入输出文件
INPUT_EXCEL = "your_file.xlsx"
OUTPUT_EXCEL = "final_processed_data.xlsx"
CODE_EMBEDDINGS_NPY = "code_str_embeddings.npy"

# 创建输出目录
Path("embeddings").mkdir(exist_ok=True)
# 创建CPG临时目录（避免重复生成，可选）
Path("./temp_cpg").mkdir(exist_ok=True)

# 初始化CodeBERT（与preprocess代码一致，用于节点编码）
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
codebert_model.eval()

# ----------------------------
# 1. 加载数据与预处理
# ----------------------------
# 读取原始Excel
df = pd.read_excel(INPUT_EXCEL)

# 提取JSON字段
results = []
for i, row in df.iterrows():
    try:
        data = json.loads(row["Data"])
        results.append({
            "component": data.get("component", ""),
            "code_str": data.get("code_str", ""),
            "Desc": data.get("Desc", ""),
            "Func": data.get("Func", ""),
            "case_id": data.get("case_id", ""),
            "test_suite": data.get("test_suite", ""),
            "case_space": data.get("case_space", ""),
            "case_purpose": data.get("case_purpose", "")
        })
    except Exception as e:
        print(f"第 {i} 行JSON解析失败: {e}")
        results.append({
            "component": "", "code_str": "", "Desc": "",
            "Func": "", "case_id": "", "test_suite": "",
            "case_space": "", "case_purpose": ""
        })

new_df = pd.DataFrame(results)
merged_df = pd.concat([df, new_df], axis=1)

# ----------------------------
# 2. 其他字段编码（复用原有逻辑）
# ----------------------------
# Sentence-BERT编码
text_model = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")


def encode_texts(texts):
    return text_model.encode([str(x) if x is not None else "" for x in texts], show_progress_bar=True)


# 文本字段编码
for col in ["Desc", "Func", "case_space", "case_purpose"]:
    embeddings = encode_texts(merged_df[col].fillna("").tolist())
    merged_df[col + "_embedding"] = [emb.tolist() for emb in embeddings]

# One-hot编码
component_onehot = pd.get_dummies(merged_df["component"], prefix="component")
case_id_onehot = pd.get_dummies(merged_df["case_id"], prefix="case_id")
test_suite_onehot = pd.get_dummies(merged_df["test_suite"], prefix="test_suite")
rule_onehot = pd.get_dummies(merged_df["rule"], prefix="rule")

# 标签处理
merged_df["false_positive"] = merged_df["status"]


# ----------------------------
# 3. code_str编码（核心：调用preprocess代码生成CPG）
# ----------------------------
class CodeEncoder:
    def __init__(self, checkpoint_path, device):
        # 加载训练好的GIN模型
        self.model = GAE_GIN_lightning.load_from_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()  # 评估模式，关闭 dropout 等
        self.device = device
        self.embedding_dim = self.model.embedding_dimension

    def _preprocess_code_to_cpg(self, code_str, sample_idx):
        """
        调用preprocess代码，将code_str转为PyG格式的CPG图
        完全复用process_sample的核心逻辑，不保存临时文件，直接返回图对象
        """
        # 1. 代码解析（与preprocess一致）
        func_name = f"func_{sample_idx}"  # 用样本索引避免函数名重复
        bytes_content = code_str.encode("utf-8")

        try:
            # 解析代码生成函数列表
            func_list = c_parser_from_serial_code(func_name, bytes_content)
            if len(func_list) < 1:
                raise ValueError(f"函数解析为空，样本索引: {sample_idx}")

            # 提取函数单元，生成CPG（AST+CFG+DDG）
            func: FunUnit = func_list[0]
            func_root = func.sast.root
            func_cpg = gen_ast_cpg(func.sast)  # 生成AST
            _, _ = cfg_build(func_cpg, func_root)  # 构建CFG
            ddg_build(func_cpg, func_root)  # 构建DDG

            # 2. 节点CodeBERT编码（与preprocess一致）
            for node_id, attrs in func_cpg.nodes(data=True):
                node: CPGNode = attrs['cpg_node']
                # 清理节点类型和token（避免特殊字符影响编码）
                node_type = node.node_type.strip('\n').replace(',', ' ')
                node_token = node.node_token.strip('\n').replace(',', ' ')

                # CodeBERT编码
                text = f"{node_type} {node_token}"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    outputs = codebert_model(**inputs)
                # 取最后一层隐藏状态的均值作为节点特征
                node_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)

                attrs['embedding'] = node_emb  # 存储节点特征
                del attrs['cpg_node']  # 删除冗余的CPGNode对象

            # 3. 边属性处理（与preprocess一致）
            for u, v, attrs in func_cpg.edges(data=True):
                edge_type = attrs['edge_type']
                # 标记边属于AST/CFG/DDG
                attrs['ast'] = edge_type[0] == '1'
                attrs['cfg'] = edge_type[1] == '1'
                attrs['ddg'] = edge_type[2] == '1'
                del attrs['edge_type']  # 删除原始边类型字段

            # 4. 转为PyG的Data格式
            torch_graph = from_networkx(func_cpg)
            # 补充model需要的字段（与DataLoader逻辑一致）
            if not hasattr(torch_graph, "connected_node_mask"):
                torch_graph.connected_node_mask = torch.ones(torch_graph.num_nodes, dtype=torch.bool)
            # 节点特征赋值给x（model默认读取x作为节点特征）
            torch_graph.x = torch.stack([attrs['embedding'] for _, attrs in func_cpg.nodes(data=True)])

            return torch_graph

        except Exception as e:
            print(f"CPG生成失败（样本索引: {sample_idx}）: {str(e)}")
            return None

    def encode_code(self, code_str, sample_idx):
        """
        完整编码流程：code_str → CPG → GIN编码 → 图向量
        sample_idx: 样本索引，用于定位错误和生成唯一函数名
        """
        # 空代码返回零向量
        if not code_str or str(code_str).strip() == "":
            return np.zeros(self.embedding_dim)

        try:
            # 步骤1：生成CPG图
            cpg_graph = self._preprocess_code_to_cpg(code_str, sample_idx)
            if cpg_graph is None:
                return np.zeros(self.embedding_dim)

            # 步骤2：GIN模型编码
            # 移动图数据到设备
            cpg_graph = cpg_graph.to(self.device)
            # 补充batch字段（单图场景，所有节点属于同一图）
            cpg_graph.batch = torch.zeros(cpg_graph.num_nodes, dtype=torch.long, device=self.device)

            with torch.no_grad():  # 关闭梯度，加速推理
                # 调用model的forward，获取图级向量（mode="predict"只返回全局向量）
                code_embedding = self.model(cpg_graph, mode="predict")

            # 转为numpy数组返回
            return code_embedding.cpu().numpy().flatten()

        except Exception as e:
            print(f"代码编码失败（样本索引: {sample_idx}）: {str(e)}")
            return np.zeros(self.embedding_dim)


# 初始化编码器
code_encoder = CodeEncoder(MODEL_CHECKPOINT, device)

# 批量编码所有code_str（带索引，方便定位错误）
print("开始编码code_str...")
code_embeddings = []
for idx, code_str in enumerate(merged_df["code_str"].fillna("")):
    # 传入样本索引，便于调试
    emb = code_encoder.encode_code(code_str, sample_idx=idx)
    code_embeddings.append(emb)

# 存储编码结果到DataFrame
merged_df["code_str_embedding"] = [emb.tolist() for emb in code_embeddings]

# ----------------------------
# 4. 整合所有特征
# ----------------------------
# 合并特征：标签 + 所有编码特征 + One-hot特征
final_df = pd.concat(
    [
        # 核心标签和向量特征
        merged_df[["false_positive", "code_str_embedding",
                   "Desc_embedding", "Func_embedding",
                   "case_space_embedding", "case_purpose_embedding"]],
        # One-hot编码特征
        component_onehot,
        case_id_onehot,
        test_suite_onehot,
        rule_onehot
    ],
    axis=1
)

# ----------------------------
# 5. 保存结果
# ----------------------------
# 保存Excel（含所有特征和标签）
final_df.to_excel(OUTPUT_EXCEL, index=False)

# 单独保存各向量特征（方便后续直接加载，无需解析Excel）
np.save(CODE_EMBEDDINGS_NPY, np.array(code_embeddings))
np.save("desc_embeddings.npy", np.array(merged_df["Desc_embedding"].tolist()))
np.save("func_embeddings.npy", np.array(merged_df["Func_embedding"].tolist()))
np.save("case_space_embeddings.npy", np.array(merged_df["case_space_embedding"].tolist()))
np.save("case_purpose_embeddings.npy", np.array(merged_df["case_purpose_embedding"].tolist()))

print(f"✅ 所有数据处理完成！")
print(f"最终特征数据: {OUTPUT_EXCEL}")
print(f"code_str编码向量: {CODE_EMBEDDINGS_NPY}")
print(
    f"特征维度: code_str({code_embeddings[0].shape[0]}) + 文本(4×384) + One-hot({len(component_onehot.columns) + len(case_id_onehot.columns) + len(test_suite_onehot.columns) + len(rule_onehot.columns)})")