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
    pkl_filename: str = Field(..., description="需要训练的 pkl 文件名")
    output_model_filename: Optional[str] = Field(
        default=None,
        description="训练后模型保存文件名，不传则自动生成"
    )
    init_ratio: float = Field(
        default=0.1,
        description="初始 labeled 集合占比"
    )
    rounds: int = Field(
        default=10,
        description="主动学习迭代轮数"
    )
    query_size: int = Field(
        default=200,
        description="每轮采样样本数"
    )
    sampling_strategy: str = Field(
        default="uncertainty_sampling",
        description="采样策略名称"
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