# data_processor.py
import pandas as pd
import json
from tqdm import tqdm
from unsupervised_train.preprocess import generate_graph_in_memory

def load_and_parse_excel(input_excel_path, onehot_fields):
    """
    加载Excel文件并解析其中的JSON字段，生成扩展特征
    :param input_excel_path: 输入Excel文件路径
    :param onehot_fields: 需要保留的离散字段列表（顺序需与训练一致）
    :return: orig_df（原始Excel数据）, merged_df（原始数据+解析后的扩展字段）
    """
    # 1. 读取最原始的 excel 数据（保留作为最终输出的基础）
    orig_df = pd.read_excel(input_excel_path)

    # 2. 按原逻辑解析 data 列中的 JSON，生成扩展字段
    results = []
    for i, row in orig_df.iterrows():
        try:
            data = json.loads(row["data"])
            raw_code = str(data.get("code_str", "")).strip()
            if raw_code.startswith('('):
                processed_code = raw_code
            elif raw_code.startswith('{'):
                processed_code = f"(){raw_code}"
            else:
                processed_code = f"(){{{raw_code}}}"

            # 构建解析结果（后续修改特征只需改这里）
            parsed_result = {
                "case_id": data.get("case_id", ""),
                "test_suite": data.get("test_suite", ""),
                "code_str": processed_code,
                "raw_code": raw_code,
                "Desc": data.get("desc", ""),
                "Func": data.get("func", ""),
                "case_spce": data.get("case_spce", ""),
                "case_purpose": data.get("case_purpose", "")
            }
            # 补充onehot字段（防止字段缺失）
            for field in onehot_fields:
                if field not in parsed_result:
                    parsed_result[field] = data.get(field, "")
            results.append(parsed_result)
        except Exception as e:
            print(f"第 {i} 行 JSON 解析失败: {e}")
            # 异常时返回空值（保证字段完整性）
            empty_result = {field: "" for field in onehot_fields}
            empty_result.append({
                "component": "", "case_id": "", "test_suite": "", "rule": "",
                "code_str": "", "raw_code": "",
                "Desc": "", "Func": "", "case_spce": "", "case_purpose": ""
            })
            results.append(empty_result)

    # 3. 合并原始数据和解析后的扩展字段
    merged_df = pd.concat([orig_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    return orig_df, merged_df

def generate_code_graphs(merged_df):
    """
    为merged_df中的code_str生成代码图（无监督编码前置步骤）
    :param merged_df: 包含code_str字段的DataFrame
    :return: 代码图列表
    """

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
    return graph_list