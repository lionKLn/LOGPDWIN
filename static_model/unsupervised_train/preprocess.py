import sys

sys.path.append('..')
import os
import io
import torch
from torch_geometric.utils.convert import from_networkx
from typing import List
from ccpg.sast.src_parser import c_parser_from_serial_code
from ccpg.sast.fun_unit import FunUnit
from ccpg.cpg.ast_constructor import gen_ast_cpg
from ccpg.cpg.cfg_constructor import cfg_build
from ccpg.cpg.ddg_constructor import ddg_build
from ccpg.cpg.cpg_node import CPGNode
from transformers import RobertaTokenizer, RobertaModel

# === 初始化 CodeBERT ===
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.eval()


def process_sample(idx, saved_dir_with_proj, sample):
    """
    对单个样本进行预处理：
    1. 解析代码 -> 生成 CPG 图
    2. 提取节点信息 -> 用 CodeBERT 编码
    3. 转换为 PyG 图结构并保存为 .pt 文件
    """
    code_str = sample['code_str']
    func_name = sample.get('func_name', f"func_{idx}")
    graph_file_path = f'{saved_dir_with_proj}/{sample["id"]}.pt'

    if os.path.exists(graph_file_path):
        return

    bytes_content = code_str.encode("utf-8")

    try:
        func_list = c_parser_from_serial_code(func_name, bytes_content)
        if len(func_list) < 1:
            print('Error: func_list should not be empty.')
            return
        func: FunUnit = func_list[0]
        func_root = func.sast.root

        func_cpg = gen_ast_cpg(func.sast)
        _, _ = cfg_build(func_cpg, func_root)
        ddg_build(func_cpg, func_root)
    except Exception as e:
        print(f"Error parsing code: {e}")
        return

    # === 对节点进行 CodeBERT 编码 ===
    for node_id, attrs in func_cpg.nodes(data=True):
        node: CPGNode = attrs['cpg_node']
        node_type = node.node_type.strip('\n').replace(',', ' ')
        node_token = node.node_token.strip('\n').replace(',', ' ')

        text = f"{node_type} {node_token}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

        attrs['embedding'] = embedding
        del attrs['cpg_node']

    # === 对边进行标注 ===
    for u, v, attrs in func_cpg.edges(data=True):
        edge_type = attrs['edge_type']
        attrs['ast'] = edge_type[0] == '1'
        attrs['cfg'] = edge_type[1] == '1'
        attrs['ddg'] = edge_type[2] == '1'
        del attrs['edge_type']

    torch_graph = from_networkx(func_cpg)
    torch.save(torch_graph, graph_file_path)
    print(f"Saved: {graph_file_path}")

