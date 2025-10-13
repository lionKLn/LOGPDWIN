import os
import pandas as pd
from tqdm import tqdm
import torch
import sys

sys.path.append('..')

from unsupervised_train.preprocess import process_sample  # 导入你之前写的函数


def load_dataset_from_xlsx(xlsx_path: str):
    """
    从 .xlsx 文件中加载数据，提取 code_str 字段。
    要求文件中至少有一列名为 'code_str'。
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"找不到文件: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    if 'code_str' not in df.columns:
        raise ValueError("Excel 文件中未找到 'code_str' 列，请确认列名正确。")

    dataset = []
    for i, row in df.iterrows():
        code_str = str(row['code_str']).strip()
        if not code_str or code_str == 'nan':
            continue
        dataset.append({
            "id": f"sample_{i}",
            "code_str": code_str
        })
    print(f"✅ 成功加载 {len(dataset)} 个样本。")
    return dataset


def build_graph_dataset(xlsx_path: str, save_dir: str):
    """
    读取 xlsx 文件，批量处理样本生成图数据 (.pt 文件)
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset = load_dataset_from_xlsx(xlsx_path)

    print(f"开始生成图数据集，保存路径：{save_dir}")
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            process_sample(idx, save_dir, sample)
        except Exception as e:
            print(f"❌ 样本 {sample['id']} 处理失败: {e}")
            continue

    print("🎯 数据集预处理完成！")


if __name__ == "__main__":
    # 例子：使用时可通过命令行运行
    xlsx_path = "../data/code_dataset.xlsx"  # 修改为你的文件路径
    save_dir = "../data/graph_dataset"  # 输出图数据保存路径

    build_graph_dataset(xlsx_path, save_dir)
