import os
import numpy as np
import pandas as pd


class SamplingService:
    """
    基于预测结果构建两个输出队列：
    1. learning queue：用于主动学习，选择最不确定样本
    2. exploitation queue：用于业务利用，选择目标标签高置信样本
    """

    def __init__(
        self,
        prediction_csv: str,
        target_label: int = 0,
    ):
        """
        :param prediction_csv: predict_pkl_to_csv 生成的结果文件
        :param target_label: 利用队列希望优先返回的目标标签
        """
        self.prediction_csv = prediction_csv
        self.target_label = target_label
        self.df = self._load_prediction_results(prediction_csv)

    @staticmethod
    def _load_prediction_results(prediction_csv: str) -> pd.DataFrame:
        if not os.path.exists(prediction_csv):
            raise FileNotFoundError(f"预测结果文件不存在: {prediction_csv}")

        df = pd.read_csv(prediction_csv)

        required_cols = ["pred_label", "prob_0", "prob_1"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"预测结果文件缺少必要列: {missing}")

        return df

    @staticmethod
    def _uncertainty_scores(prob_1: np.ndarray) -> np.ndarray:
        """
        二分类不确定性分数：
        越接近 0.5，越不确定。
        这里返回的是 abs(p-0.5)，值越小越不确定。
        """
        return np.abs(prob_1 - 0.5)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        简单归一化到 [0,1]，用于可视化或后续查看。
        若所有值相同，则返回全0。
        """
        if len(scores) == 0:
            return scores
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min < 1e-12:
            return np.zeros_like(scores, dtype=float)
        return (scores - s_min) / (s_max - s_min)

    def build_learning_queue(
        self,
        n_samples: int = 50,
    ) -> pd.DataFrame:
        """
        构建用于学习的样本队列：
        选择最不确定的样本（最接近决策边界的样本）

        :param n_samples: 需要输出的样本数
        :return: learning queue DataFrame
        """
        df = self.df.copy()
        prob_1 = df["prob_1"].values
        uncertainty_scores = self._uncertainty_scores(prob_1)

        df["uncertainty_score"] = uncertainty_scores
        df["uncertainty_rank_score"] = 1.0 - self._normalize_scores(uncertainty_scores)
        # 越小越不确定，因此升序
        df = df.sort_values(by="uncertainty_score", ascending=True).reset_index(drop=False)
        df.rename(columns={"index": "original_index"}, inplace=True)

        learning_df = df.head(n_samples).copy()
        learning_df["queue_type"] = "learning"
        learning_df["sampling_strategy"] = "uncertainty"

        return learning_df

    def build_exploitation_queue(
        self,
        n_samples: int = 100,
        target_label: int = None,
        exclude_indices: set = None,
    ) -> pd.DataFrame:
        """
        构建用于利用的样本队列：
        选择目标标签高置信度样本

        :param n_samples: 需要输出的样本数
        :param target_label: 目标标签；None 时使用 self.target_label
        :param exclude_indices: 需要排除的 original_index 集合，例如学习队列中的样本
        :return: exploitation queue DataFrame
        """
        if target_label is None:
            target_label = self.target_label

        prob_col = f"prob_{target_label}"
        if prob_col not in self.df.columns:
            raise ValueError(f"预测结果文件中不存在列: {prob_col}")

        df = self.df.copy().reset_index(drop=False)
        df.rename(columns={"index": "original_index"}, inplace=True)

        if exclude_indices is not None and len(exclude_indices) > 0:
            df = df[~df["original_index"].isin(exclude_indices)].copy()

        df["target_confidence"] = df[prob_col].values
        df["target_label_for_exploitation"] = target_label
        # 置信度越高越优先，因此降序
        df = df.sort_values(by="target_confidence", ascending=False).reset_index(drop=True)

        exploitation_df = df.head(n_samples).copy()
        exploitation_df["queue_type"] = "exploitation"
        exploitation_df["sampling_strategy"] = f"high_confidence_label_{target_label}"

        return exploitation_df

    def generate_two_queues(
        self,
        learning_size: int = 50,
        exploitation_size: int = 100,
        target_label: int = None,
        allow_overlap: bool = False,
    ):
        """
        同时生成学习队列和利用队列

        :param learning_size: 学习队列大小
        :param exploitation_size: 利用队列大小
        :param target_label: 利用队列目标标签
        :param allow_overlap: 是否允许两个队列出现同一样本
        :return: learning_df, exploitation_df
        """
        learning_df = self.build_learning_queue(n_samples=learning_size)

        exclude_indices = None
        if not allow_overlap:
            exclude_indices = set(learning_df["original_index"].tolist())

        exploitation_df = self.build_exploitation_queue(
            n_samples=exploitation_size,
            target_label=target_label,
            exclude_indices=exclude_indices,
        )

        return learning_df, exploitation_df


def save_queue(df: pd.DataFrame, output_csv: str):
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"队列文件已保存: {output_csv}")
    print(df.head())


def generate_sampling_outputs(
    prediction_csv: str,
    learning_output_csv: str,
    exploitation_output_csv: str,
    learning_size: int = 50,
    exploitation_size: int = 100,
    target_label: int = 0,
    allow_overlap: bool = False,
):
    """
    对外工程入口：
    输入预测结果 CSV，输出 learning queue 和 exploitation queue
    """
    service = SamplingService(
        prediction_csv=prediction_csv,
        target_label=target_label
    )

    learning_df, exploitation_df = service.generate_two_queues(
        learning_size=learning_size,
        exploitation_size=exploitation_size,
        target_label=target_label,
        allow_overlap=allow_overlap,
    )

    save_queue(learning_df, learning_output_csv)
    save_queue(exploitation_df, exploitation_output_csv)

    print("\n采样输出完成。")
    print(f"- 学习队列数量: {len(learning_df)}")
    print(f"- 利用队列数量: {len(exploitation_df)}")
    print(f"- 目标标签: {target_label}")
    print(f"- 是否允许重叠: {allow_overlap}")


if __name__ == "__main__":
    generate_sampling_outputs(
        prediction_csv="inference_results.csv",
        learning_output_csv="learning_queue.csv",
        exploitation_output_csv="exploitation_queue.csv",
        learning_size=50,
        exploitation_size=100,
        target_label=0,      # 例如标签0表示有缺陷
        allow_overlap=False  # 默认不允许与学习队列重复
    )