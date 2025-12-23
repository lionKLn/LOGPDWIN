import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import traceback
import uvicorn

from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Batch
from sklearn.preprocessing import OneHotEncoder

import sys

sys.path.append('..')
from supervised_train.model import LogClassifier
from unsupervised_train.model import GAE_GIN
from unsupervised_train.preprocess import generate_graph_in_memory

# Pydantic模型定义
class ModelConfig(BaseModel):
    unsupervised_model_path: str = Field(default='logs/pdg/2025-05-20_14-30-00/best_pdg.pt', description="无监督图模型路径")
    classifier_model_path: str = Field(default='best_log_classifier.pt', description="分类器模型路径")
    onehot_encoder_path: str = Field(default='onehot_encoder.pkl', description="One-hot编码器路径")
    onehot_features_path: str = Field(default='onehot_feature_names.npy', description="One-hot特征名称路径")
    text_model_path: str = Field(default='./models/paraphrase-multilingual-MiniLM-L12-v2', description="文本模型路径")

class PredictionRequest(BaseModel):
    component: str = Field(..., description="组件名称")
    case_id: str = Field(..., description="用例ID")
    test_suite: str = Field(..., description="测试套件")
    rule: str = Field(..., description="规则名称")
    code_str: str = Field(..., description="代码字符串")
    desc: Optional[str] = Field("", description="描述信息")
    func: Optional[str] = Field("", description="函数名称")
    case_spce: Optional[str] = Field("", description="用例规格")
    case_purpose: Optional[str] = Field("", description="用例目的")

class PredictionResponse(BaseModel):
    pred_label: int = Field(..., description="预测标签")
    prob_0: float = Field(..., description="类别0的概率")
    prob_1: float = Field(..., description="类别1的概率")
    confidence: float = Field(..., description="置信度")
    status: str = Field(..., description="状态")
    error: Optional[str] = Field(None, description="错误信息（如果有）")

class BatchPredictionRequest(BaseModel):
    batch: List[PredictionRequest] = Field(..., description="批量预测请求数据")

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse] = Field(..., description="预测结果列表")
    count: int = Field(..., description="结果数量")
    status: str = Field(..., description="状态")

class HealthResponse(BaseModel):
    status: str = Field(..., description="健康状态")
    timestamp: str = Field(..., description="时间戳")
    device: str = Field(..., description="设备信息")

class ModelStatusResponse(BaseModel):
    graph_model: bool = Field(..., description="图模型是否加载")
    text_model: bool = Field(..., description="文本模型是否加载")
    classifier_model: bool = Field(..., description="分类器模型是否加载")
    encoder: bool = Field(..., description="编码器是否加载")
    device: str = Field(..., description="设备信息")

class LoadModelsResponse(BaseModel):
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    device: str = Field(..., description="设备信息")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="错误信息")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelInferenceService:
    def __init__(self):
        self.device = torch.device("npu:4" if torch.npu.is_available() else "cpu")
        self.embedding_dim = 256
        self.hidden_dim = 128
        
        self.graph_model = None
        self.text_model = None
        self.tokenizer = None
        self.classifier_model = None
        self.encoder = None
        self.encoder_columns = None
        
        self.onehot_fields = ["component", "case_id", "test_suite", "rule"]
        self.text_columns = ["desc", "func", "case_spce", "case_purpose"]
        
        logger.info(f"Using device: {self.device}")
    
    def load_models(self, config: ModelConfig):
        """加载所有必要的模型和编码器"""
        try:
            logger.info("开始加载模型...")
            
            if not os.path.exists(config.onehot_encoder_path) or not os.path.exists(config.onehot_features_path):
                raise FileNotFoundError(f"编码器文件不存在: {config.onehot_encoder_path} 或 {config.onehot_features_path}")
            
            self.encoder = joblib.load(config.onehot_encoder_path)
            self.encoder_columns = np.load(config.onehot_features_path, allow_pickle=True).tolist()
            
            logger.info("✅ One-hot编码器加载成功")
            
            self.graph_model = GAE_GIN(in_channels=768, out_channels=768, device=self.device).to(self.device)
            if os.path.exists(config.unsupervised_model_path):
                self.graph_model.load_state_dict(torch.load(config.unsupervised_model_path, map_location=self.device))
                self.graph_model.eval()
                logger.info("✅ 图神经网络模型加载成功")
            else:
                logger.warning(f"图神经网络模型文件不存在: {config.unsupervised_model_path}")
            
            if os.path.exists(config.text_model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_path)
                self.text_model = AutoModel.from_pretrained(config.text_model_path).to(self.device)
                self.text_model.eval()
                logger.info("✅ 文本编码模型加载成功")
            else:
                logger.warning(f"文本编码模型路径不存在: {config.text_model_path}")
            
            if os.path.exists(config.classifier_model_path):
                self.classifier_model = LogClassifier(input_dim=1935, hidden_dim=self.hidden_dim)
                self.classifier_model.load_state_dict(torch.load(config.classifier_model_path, map_location=self.device))
                self.classifier_model.to(self.device)
                self.classifier_model.eval()
                logger.info("✅ 分类器模型加载成功")
            else:
                logger.warning(f"分类器模型文件不存在: {config.classifier_model_path}")
            
            logger.info("✅ 所有模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def preprocess_code(self, code_str: str) -> str:
        """预处理代码字符串"""
        raw_code = str(code_str).strip()
        if raw_code.startswith('('):
            return raw_code
        elif raw_code.startswith('{'):
            return f"(){raw_code}"
        else:
            return f"(){{{raw_code}}}"
    
    def encode_graph(self, code_str: str, func_name: str) -> Optional[torch.Tensor]:
        """编码代码图为向量"""
        try:
            processed_code = self.preprocess_code(code_str)
            torch_graph = generate_graph_in_memory(code_str=processed_code, func_name=func_name)
            
            if torch_graph is None:
                return torch.zeros(self.embedding_dim, device=self.device)
            
            batch = Batch.from_data_list([torch_graph]).to(self.device)
            
            with torch.no_grad():
                if self.graph_model is not None:
                    embedding = self.graph_model.forward(batch, mode="predict")
                    return embedding[0] if len(embedding) > 0 else torch.zeros(self.embedding_dim, device=self.device)
                else:
                    return torch.zeros(self.embedding_dim, device=self.device)
                    
        except Exception as e:
            logger.error(f"图编码失败: {str(e)}")
            return torch.zeros(self.embedding_dim, device=self.device)
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling for text embeddings"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode_text(self, text: str) -> List[float]:
        """编码文本为向量"""
        try:
            if not text or self.text_model is None or self.tokenizer is None:
                return [0.0] * 384
            
            text = str(text).strip()
            if not text:
                return [0.0] * 384
            
            encoded = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
                sentence_embedding = self.mean_pooling(outputs, attention_mask)
                return sentence_embedding.cpu().tolist()[0]
                
        except Exception as e:
            logger.error(f"文本编码失败: {str(e)}")
            raise e    
    def encode_onehot(self, data: Dict[str, str]) -> List[float]:
        """One-hot编码离散特征"""
        if self.encoder is None or self.encoder_columns is None:
            raise FileNotFoundError("❌ 未找到编码器文件，请确保 onehot_encoder.pkl 和 onehot_feature_names.npy 存在")

        onehot_input = pd.DataFrame([data])[self.onehot_fields].fillna("").astype(str)

        try:
            onehot_encoded = self.encoder.transform(onehot_input)
        except ValueError as e:
            raise ValueError(f"❌ 编码失败：输入字段与训练时不一致，请检查 ONEHOT_FIELDS。错误详情：{e}")

        onehot_df = pd.DataFrame(onehot_encoded, columns=self.encoder.get_feature_names_out(self.onehot_fields))

        for col in self.encoder_columns:
            if col not in onehot_df.columns:
                onehot_df[col] = 0
        onehot_df = onehot_df[self.encoder_columns]

        return onehot_df.iloc[0].tolist()
    
    def merge_features(self, code_embedding: List[float], text_embeddings: Dict[str, List[float]], 
                      onehot_embedding: List[float]) -> List[float]:
        """合并所有特征"""
        text_embs = []
        for col in self.text_columns:
            text_embs.extend(text_embeddings.get(col, [0.0] * 384))
        
        return code_embedding + text_embs + onehot_embedding
    
    def predict_single(self, data: PredictionRequest) -> PredictionResponse:
        """对单个样本进行预测"""
        try:
            logger.info(f"开始处理样本: {data.case_id}")
            
            code_str = data.code_str
            func_name = f"func_{datetime.now().timestamp()}"
            
            code_embedding = self.encode_graph(code_str, func_name)
            code_embedding_cpu = code_embedding.cpu().tolist()
            
            text_embeddings = {}
            for col in self.text_columns:
                text_value = getattr(data, col.lower(), getattr(data, col, ""))
                text_embeddings[col] = self.encode_text(text_value)
            
            onehot_data = {}
            for field in self.onehot_fields:
                onehot_data[field] = getattr(data, field, "")
            
            onehot_embedding = self.encode_onehot(onehot_data)
            
            merged_features = self.merge_features(code_embedding_cpu, text_embeddings, onehot_embedding)
            
            feature_tensor = torch.tensor([merged_features], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                if self.classifier_model is not None:
                    outputs = self.classifier_model(feature_tensor)
                    probs = F.softmax(outputs, dim=1).cpu().numpy()
                    pred = int(np.argmax(probs, axis=1)[0])
                    prob_0 = float(probs[0][0])
                    prob_1 = float(probs[0][1])
                else:
                    logger.error("分类器模型未加载")
                    raise ValueError("分类器模型未加载")
            
            result = PredictionResponse(
                pred_label=pred,
                prob_0=prob_0,
                prob_1=prob_1,
                confidence=max(prob_0, prob_1),
                status='success'
            )
            
            logger.info(f"预测完成: label={pred}, confidence={result.confidence}")
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise e
    
    def predict_batch(self, batch_data: List[PredictionRequest]) -> List[PredictionResponse]:
        """批量预测"""
        results = []
        for i, data in enumerate(batch_data):
            logger.info(f"处理批量样本 {i+1}/{len(batch_data)}")
            result = self.predict_single(data)
            results.append(result)
        return results

# FastAPI应用
app = FastAPI(
    title="模型推理API服务",
    description="基于图神经网络和文本编码的机器学习模型推理服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ModelInferenceService()

@app.on_event("startup")
async def startup_event():
    """应用启动时加载默认模型"""
    logger.info("启动模型推理API服务...")
    
    # 设置模型配置
    model_config = ModelConfig(
        
    )
    
    try:
        service.load_models(model_config)
        logger.info("✅ 默认模型加载完成")
    except Exception as e:
        logger.warning(f"默认模型加载失败: {str(e)}")
        logger.info("请通过 /load_models 接口手动加载模型")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status='healthy',
        timestamp=datetime.now().isoformat(),
        device=str(service.device)
    )

@app.post("/load_models", response_model=LoadModelsResponse)
async def load_models(config: ModelConfig):
    """加载模型接口"""
    try:
        service.load_models(config)
        return LoadModelsResponse(
            status='success',
            message='Models loaded successfully',
            device=str(service.device)
        )
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型加载失败: {str(e)}"
        )

@app.post("/static_predict", response_model=PredictionResponse)
async def static_predict(request: PredictionRequest):
    """单个样本预测接口"""
    try:
        result = service.predict_single(request)
        return result
    except Exception as e:
        logger.error(f"预测接口错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败: {str(e)}"
        )

@app.post("/static_predict_batch", response_model=BatchPredictionResponse)
async def static_predict_batch(request: BatchPredictionRequest):
    """批量预测接口"""
    try:
        results = service.predict_batch(request.batch)
        return BatchPredictionResponse(
            results=results,
            count=len(results),
            status='success'
        )
    except Exception as e:
        logger.error(f"批量预测接口错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量预测失败: {str(e)}"
        )

@app.get("/models/status", response_model=ModelStatusResponse)
async def models_status():
    """模型状态检查"""
    return ModelStatusResponse(
        graph_model=service.graph_model is not None,
        text_model=service.text_model is not None,
        classifier_model=service.classifier_model is not None,
        encoder=service.encoder is not None,
        device=str(service.device)
    )

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "模型推理API服务",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
