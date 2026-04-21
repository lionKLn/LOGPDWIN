from typing import Optional, Literal
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[dict] = None


class InferRequest(BaseModel):
    pkl_filename: str = Field(..., description="需要推理的 pkl 文件名")
    model_filename: str = Field(..., description="用于推理的模型文件名")
    output_csv_filename: Optional[str] = Field(
        default=None,
        description="推理结果保存的 csv 文件名，不传则自动生成"
    )


class TrainRequest(BaseModel):
    # =========================
    # 基础输入
    # =========================
    pkl_filename: str = Field(
        ...,
        description="需要训练的 pkl 文件名"
    )

    output_model_filename: Optional[str] = Field(
        default=None,
        description="训练后模型保存文件名，不传则自动生成"
    )

    # =========================
    # 数据划分相关（load_and_split_active_learning）
    # =========================
    test_size: float = Field(
        default=0.2,
        description="测试集比例（固定 test 集）"
    )

    init_ratio: float = Field(
        default=0.1,
        description="初始 labeled 集合占比（主动学习起始标注量）"
    )

    seed: int = Field(
        default=42,
        description="随机种子"
    )

    dedup_by_id: bool = Field(
        default=True,
        description="是否根据 id 去重（避免重复样本）"
    )

    # =========================
    # 主动学习参数（active_learning_loop）
    # =========================
    rounds: int = Field(
        default=10,
        description="主动学习迭代轮数"
    )

    query_size: int = Field(
        default=200,
        description="每轮从 pool 中采样的样本数量"
    )

    sampling_strategy: str = Field(
        default="uncertainty_sampling",
        description="采样策略（如 uncertainty_sampling / entropy / margin 等）"
    )

    # =========================
    # （可选）模型训练参数（扩展用）
    # =========================
    hidden_dim: int = Field(
        default=128,
        description="模型隐藏层维度（LogClassifier）"
    )

    epochs: int = Field(
        default=10,
        description="每轮训练的 epoch 数"
    )

    learning_rate: float = Field(
        default=5e-4,
        description="学习率"
    )


class SamplingRequest(BaseModel):
    csv_filename: str = Field(..., description="需要采样的推理结果 csv 文件名")
    learning_output_filename: Optional[str] = Field(
        default=None,
        description="learning queue 输出文件名"
    )
    exploitation_output_filename: Optional[str] = Field(
        default=None,
        description="exploitation queue 输出文件名"
    )
    learning_size: int = Field(
        default=50,
        description="learning queue 样本数"
    )
    exploitation_size: int = Field(
        default=100,
        description="exploitation queue 样本数"
    )
    target_label: int = Field(
        default=0,
        description="利用队列目标标签"
    )
    allow_overlap: bool = Field(
        default=False,
        description="learning 和 exploitation 两个队列是否允许重叠"
    )