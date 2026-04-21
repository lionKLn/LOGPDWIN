import shutil
from uuid import uuid4
from typing import Literal, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)

from config import UPLOAD_DIR, PKL_DIR, CSV_DIR, MODEL_DIR, ENCODER_DIR
from schemas import BaseResponse, InferRequest, TrainRequest, SamplingRequest

from supervised_train.encode.encode_to_pkl import encode_excel_to_pkl
from supervised_train.predict.predict_from_pkl import predict_pkl_to_csv
from supervised_train.active_learning.data_loader import load_and_split_active_learning
from supervised_train.active_learning.active_loop import active_learning_loop
from supervised_train.sample.sampling_service import generate_sampling_outputs


app = FastAPI(
    title="Active Learning Service",
    version="0.1.0",
    description="用于编码、推理、训练与采样的服务端接口"
)


def ensure_file_exists(file_path: Path, file_desc: str = "文件"):
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{file_desc}不存在: {file_path.name}")


@app.get("/")
def root():
    return {
        "message": "服务已启动",
        "endpoints": [
            "/encode",
            "/infer",
            "/train",
            "/sample"
        ]
    }


# ==========================================
# 1. 编码接口
# ==========================================
@app.post("/encode", response_model=BaseResponse)
async def encode_file(
    file: UploadFile = File(..., description="上传的 Excel 文件"),
    mode: Literal["train", "infer"] = Form(..., description="编码模式：train 或 infer"),
    encoder_tag: Optional[str] = Form(
        default=None,
        description="编码器标识。train 模式可不传，infer 模式必须传，表示使用哪套编码器"
    )
):
    """
    传入一个 excel 文件和 mode：
    - train: 生成训练用 pkl，并额外生成一套专属编码器文件
    - infer: 生成推理用 pkl，必须指定要使用的编码器
    """
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="仅支持 Excel 文件（.xlsx / .xls）")

    unique_id = uuid4().hex
    upload_filename = f"{unique_id}_{file.filename}"
    upload_path = UPLOAD_DIR / upload_filename

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 输出 pkl 文件名
    output_pkl_filename = f"{Path(file.filename).stem}_{mode}_{unique_id}.pkl"
    output_pkl_path = PKL_DIR / output_pkl_filename

    # ===== 编码器文件绑定逻辑 =====
    # train 模式：自动生成一套新的 encoder_tag
    # infer 模式：必须提供 encoder_tag，以保证和模型对应
    if mode == "train":
        final_encoder_tag = encoder_tag or unique_id
    else:
        if not encoder_tag:
            raise HTTPException(
                status_code=400,
                detail="infer 模式必须提供 encoder_tag，用于指定与模型对应的编码器文件"
            )
        final_encoder_tag = encoder_tag

    onehot_encoder_filename = f"{final_encoder_tag}_onehot_encoder.pkl"
    onehot_feature_names_filename = f"{final_encoder_tag}_onehot_feature_names.npy"

    onehot_encoder_path = ENCODER_DIR / onehot_encoder_filename
    onehot_feature_names_path = ENCODER_DIR / onehot_feature_names_filename

    # infer 模式下，编码器必须已经存在
    if mode == "infer":
        if not onehot_encoder_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"infer 模式所需编码器不存在: {onehot_encoder_filename}"
            )
        if not onehot_feature_names_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"infer 模式所需编码器特征文件不存在: {onehot_feature_names_filename}"
            )

    # 下面这些路径相对于 static_model 根目录
    unsupervised_model_path = PROJECT_ROOT / "pdg_model" / "best_pdg.pt"
    text_model_path = PROJECT_ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"

    if not unsupervised_model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"无监督图编码模型不存在: {unsupervised_model_path}"
        )

    if not text_model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"文本编码模型目录不存在: {text_model_path}"
        )

    try:
        encode_excel_to_pkl(
            input_excel=str(upload_path),
            output_pkl=str(output_pkl_path),
            unsupervised_model_path=str(unsupervised_model_path),
            text_model_path=str(text_model_path),
            onehot_encoder_path=str(onehot_encoder_path),
            onehot_feature_names_path=str(onehot_feature_names_path),
            embedding_dim=256,
            mode=mode
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"编码失败: {str(e)}")

    return BaseResponse(
        message="编码完成",
        data={
            "mode": mode,
            "uploaded_excel": upload_filename,
            "saved_excel_path": str(upload_path),
            "output_pkl_filename": output_pkl_filename,
            "output_pkl_path": str(output_pkl_path),
            "encoder_tag": final_encoder_tag,
            "onehot_encoder_filename": onehot_encoder_filename,
            "onehot_encoder_path": str(onehot_encoder_path),
            "onehot_feature_names_filename": onehot_feature_names_filename,
            "onehot_feature_names_path": str(onehot_feature_names_path),
            "note": "后续训练出的模型需要与该 encoder_tag 绑定；推理时必须使用相同 encoder_tag"
        }
    )


# ==========================================
# 2. 推理接口
# ==========================================
@app.post("/infer", response_model=BaseResponse)
async def infer_from_pkl(req: InferRequest):
    """
    传入 pkl 文件名和模型文件名，服务端进行推理，并保存结果 csv。
    """
    pkl_path = PKL_DIR / req.pkl_filename
    ensure_file_exists(pkl_path, "pkl 文件")

    model_path = MODEL_DIR / req.model_filename
    ensure_file_exists(model_path, "模型文件")

    output_csv_filename = req.output_csv_filename
    if not output_csv_filename:
        output_csv_filename = f"{Path(req.pkl_filename).stem}_{Path(req.model_filename).stem}_inference_results.csv"

    output_csv_path = CSV_DIR / output_csv_filename

    try:
        predict_pkl_to_csv(
            input_pkl=str(pkl_path),
            model_path=str(model_path),
            output_csv=str(output_csv_path),
            hidden_dim=128
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

    return BaseResponse(
        message="推理完成",
        data={
            "input_pkl_filename": req.pkl_filename,
            "input_pkl_path": str(pkl_path),
            "model_filename": req.model_filename,
            "model_path": str(model_path),
            "output_csv_filename": output_csv_filename,
            "output_csv_path": str(output_csv_path)
        }
    )


# ==========================================
# 3. 训练接口
# ==========================================
@app.post("/train", response_model=BaseResponse)
async def train_from_pkl(req: TrainRequest):
    """
    传入 pkl 文件名，服务端进行主动学习训练，并保存模型文件。
    """
    pkl_path = PKL_DIR / req.pkl_filename
    ensure_file_exists(pkl_path, "训练 pkl 文件")

    output_model_filename = req.output_model_filename
    if not output_model_filename:
        output_model_filename = f"{Path(req.pkl_filename).stem}_trained_model.pt"

    output_model_path = MODEL_DIR / output_model_filename

    try:
        # 1. 直接从 pkl 划分出：
        # labeled / pool / test
        X_labeled, X_pool, y_labeled, y_pool, X_test, y_test = load_and_split_active_learning(
            pkl_path=str(pkl_path),
            test_size=req.test_size,
            init_ratio=req.init_ratio,
            seed=req.seed,
            dedup_by_id=req.dedup_by_id
        )

        print(f"初始 labeled: {len(X_labeled)}, pool: {len(X_pool)}")

        # 3. 主动学习训练
        model, history = active_learning_loop(
            X_labeled,
            X_pool,
            y_labeled,
            y_pool,
            rounds=req.rounds,
            query_size=req.query_size,
            sampling_strategy=req.sampling_strategy,
            mode_save_path=str(output_model_path)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

    return BaseResponse(
        message="训练完成",
        data={
            "input_pkl_filename": req.pkl_filename,
            "input_pkl_path": str(pkl_path),
            "output_model_filename": output_model_filename,
            "output_model_path": str(output_model_path),
            "init_ratio": req.init_ratio,
            "rounds": req.rounds,
            "query_size": req.query_size,
            "sampling_strategy": req.sampling_strategy,
            "history": history
        }
    )


# ==========================================
# 4. 采样接口
# ==========================================
# ==========================================
# 4. 采样接口
# ==========================================
@app.post("/sample", response_model=BaseResponse)
async def sample_from_csv(req: SamplingRequest):
    """
    传入推理结果 csv 文件名，服务端进行采样，
    输出 learning queue 和 exploitation queue 两个 csv。
    """
    csv_path = CSV_DIR / req.csv_filename
    ensure_file_exists(csv_path, "推理结果 csv 文件")

    learning_output_filename = req.learning_output_filename
    if not learning_output_filename:
        learning_output_filename = f"{Path(req.csv_filename).stem}_learning_queue.csv"

    exploitation_output_filename = req.exploitation_output_filename
    if not exploitation_output_filename:
        exploitation_output_filename = f"{Path(req.csv_filename).stem}_exploitation_queue.csv"

    learning_output_path = CSV_DIR / learning_output_filename
    exploitation_output_path = CSV_DIR / exploitation_output_filename

    try:
        generate_sampling_outputs(
            prediction_csv=str(csv_path),
            learning_output_csv=str(learning_output_path),
            exploitation_output_csv=str(exploitation_output_path),
            learning_size=req.learning_size,
            exploitation_size=req.exploitation_size,
            target_label=req.target_label,
            allow_overlap=req.allow_overlap
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"采样失败: {str(e)}")

    return BaseResponse(
        message="采样完成",
        data={
            "input_csv_filename": req.csv_filename,
            "input_csv_path": str(csv_path),
            "learning_output_filename": learning_output_filename,
            "learning_output_path": str(learning_output_path),
            "exploitation_output_filename": exploitation_output_filename,
            "exploitation_output_path": str(exploitation_output_path),
            "learning_size": req.learning_size,
            "exploitation_size": req.exploitation_size,
            "target_label": req.target_label,
            "allow_overlap": req.allow_overlap
        }
    )


# 启动使用命令uvicorn app:app --reload
