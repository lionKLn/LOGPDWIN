import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import logging
from typing import Optional, Union
import argparse
from pathlib import Path
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceProcessor:
    def __init__(
            self,
            encoder_path: str = 'encoder.pkl',
            encoder_columns_path: str = 'encoder_columns.npy',
            codebert_path: str = "./codebert",
            device: Optional[str] = None
    ):
        """
        初始化特征处理器
        :param encoder_path: OneHotEncoder保存路径
        :param encoder_columns_path: 编码列名保存路径
        :param codebert_path: CodeBERT模型路径
        :param device: 指定设备 (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.onehot_fields = ['oracle_name', 'sut.component', 'sut.component_set', 'sut.module']
        self.codebert_dim = 768

        # 加载预训练组件
        self._load_artifacts(encoder_path, encoder_columns_path, codebert_path)

    def _load_artifacts(self, encoder_path: str, encoder_columns_path: str, codebert_path: str):
        """加载所有预训练组件"""
        try:
            logger.info("🔄 正在加载预训练组件...")

            # 验证文件存在
            if not Path(encoder_path).exists():
                raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
            if not Path(encoder_columns_path).exists():
                raise FileNotFoundError(f"Encoder columns file not found: {encoder_columns_path}")

            # 加载OneHot编码器
            self.onehot_encoder = joblib.load(encoder_path)
            self.expected_onehot_columns = np.load(encoder_columns_path, allow_pickle=True)
            logger.info(f"✅ 成功加载OneHotEncoder，特征数: {len(self.expected_onehot_columns)}")

            # 加载CodeBERT模型
            if not Path(codebert_path).exists():
                raise FileNotFoundError(f"CodeBERT path not found: {codebert_path}")

            start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(codebert_path)
            self.codebert_model = AutoModel.from_pretrained(codebert_path).to(self.device)
            self.codebert_model.eval()
            load_time = time.time() - start_time
            logger.info(f"✅ 成功加载CodeBERT模型 (设备: {self.device}, 耗时: {load_time:.2f}s)")

        except Exception as e:
            logger.error(f"❌ 加载组件失败: {str(e)}")
            raise

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        验证并预处理输入数据
        :return: 处理后的DataFrame
        """
        # 检查必需字段
        required_cols = self.onehot_fields + ['api_ut', 'tags']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"❌ 缺少必要列: {missing_cols}")

        # 复制数据避免修改原始数据
        processed_df = df.copy()

        # 填充缺失值
        for col in self.onehot_fields:
            if processed_df[col].isna().any():
                na_count = processed_df[col].isna().sum()
                logger.warning(f"⚠️ 在列 {col} 中发现 {na_count} 个NA值，将填充为'missing'")
                processed_df[col] = processed_df[col].fillna('missing')

        return processed_df

    def _encode_text_batch(self, texts: pd.Series, batch_size: int = 32) -> np.ndarray:
        """
        批量编码文本字段
        :param texts: 文本序列
        :param batch_size: 批处理大小
        :return: numpy数组 (n_samples, 768)
        """
        embeddings = []
        texts = texts.fillna('').astype(str).tolist()
        total_batches = (len(texts) // batch_size + (1 if len(texts) % batch_size else 0)

                         for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        try:
            start_time = time.time()
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.codebert_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

            batch_time = time.time() - start_time
            logger.debug(f"🔧 处理批次 {batch_num}/{total_batches} (大小: {len(batch)}), 耗时: {batch_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ 处理批次 {batch_num} 失败: {str(e)}")
            # 失败时返回零向量
            embeddings.append(np.zeros((len(batch), self.codebert_dim)))

        return np.vstack(embeddings)

    def process_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        完整特征处理流水线
        :param df: 输入DataFrame
        :return: 处理后的特征矩阵
        """
        logger.info("🛠️ 开始特征处理流程...")

        # 1. 数据验证与预处理
        processed_df = self._validate_input(df)
        logger.info(f"📊 输入数据形状: {processed_df.shape}")

        # 2. OneHot编码
        try:
            logger.info("🔠 正在进行OneHot编码...")
            onehot_data = self.onehot_encoder.transform(processed_df[self.onehot_fields])
            onehot_df = pd.DataFrame(
                onehot_data,
                columns=self.onehot_encoder.get_feature_names_out()
            )
            # 对齐特征列
            onehot_df = onehot_df.reindex(columns=self.expected_onehot_columns, fill_value=0)
            logger.info(f"✅ OneHot编码完成，形状: {onehot_df.shape}")
        except Exception as e:
            logger.error(f"❌ OneHot编码失败: {str(e)}")
            raise

        # 3. 文本编码
        try:
            logger.info("📝 正在编码文本字段(api_ut)...")
            api_ut_embeds = self._encode_text_batch(processed_df['api_ut'])
            logger.info("🏷️ 正在编码文本字段(tags)...")
            tag_embeds = self._encode_text_batch(processed_df['tags'])
            logger.info(f"✅ 文本编码完成，形状: api_ut={api_ut_embeds.shape}, tags={tag_embeds.shape}")
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {str(e)}")
            raise

        # 4. 合并特征
        X = np.hstack([onehot_df.values, api_ut_embeds, tag_embeds])
        logger.info(f"🎉 特征处理完成! 最终特征矩阵形状: {X.shape}")

        return X


def load_data(file_path: str) -> pd.DataFrame:
    """加载输入数据文件"""
    try:
        logger.info(f"📂 正在加载数据文件: {file_path}")

        # 支持多种文件格式
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("不支持的文件格式，请使用CSV/Parquet/JSON")

        logger.info(f"✅ 成功加载数据，行数: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"❌ 加载数据文件失败: {str(e)}")
        raise


def main():
    """主处理流程"""
    parser = argparse.ArgumentParser(description='测试数据特征处理器')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径(CSV/Parquet/JSON)')
    parser.add_argument('--output', type=str, default='output_features.npy', help='特征输出路径')
    parser.add_argument('--encoder', type=str, default='encoder.pkl', help='OneHotEncoder路径')
    parser.add_argument('--encoder-cols', type=str, default='encoder_columns.npy', help='编码列名路径')
    parser.add_argument('--codebert', type=str, default='./codebert', help='CodeBERT模型路径')
    args = parser.parse_args()

    try:
        # 1. 加载数据
        df = load_data(args.input)

        # 2. 初始化处理器
        processor = InferenceProcessor(
            encoder_path=args.encoder,
            encoder_columns_path=args.encoder_cols,
            codebert_path=args.codebert
        )

        # 3. 处理特征
        start_time = time.time()
        features = processor.process_features(df)
        process_time = time.time() - start_time

        # 4. 保存结果
        np.save(args.output, features)
        logger.info(f"💾 特征已保存到 {args.output}，总耗时: {process_time:.2f}秒")

    except Exception as e:
        logger.error(f"🔥 处理失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()