# 模型推理API服务文档

## 概述

这是一个基于FastAPI的现代化RESTful API服务器，提供机器学习模型的推理服务。该服务支持代码分析、文本编码、图神经网络编码和分类预测功能。FastAPI提供了类型安全、自动文档生成和高性能异步处理能力。

## 环境要求

- Python 3.7+
- PyTorch
- FastAPI
- Uvicorn
- Pydantic
- transformers
- torch-geometric
- scikit-learn
- pandas
- numpy

## 安装依赖

```bash
pip install fastapi uvicorn pydantic torch transformers torch-geometric scikit-learn pandas numpy joblib
```

## 启动服务

```bash
python server.py
```

服务将在 `http://localhost:8000` 启动

## API端点

### 0. 根路径

**GET** `/`

获取API基本信息和文档链接。

**响应示例:**
```json
{
    "message": "模型推理API服务",
    "version": "1.0.0",
    "docs": "/docs",
    "redoc": "/redoc"
}
```

### 1. API文档

**GET** `/docs` (Swagger UI)
**GET** `/redoc` (ReDoc)

FastAPI自动生成的API文档，包含所有端点的详细信息和交互式测试界面。

### 2. 健康检查

**GET** `/health`

检查服务是否正常运行。

**响应示例:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00.000000",
    "device": "cpu"
}
```

### 3. 模型状态检查

**GET** `/models/status`

检查各个模型的加载状态。

**响应示例:**
```json
{
    "graph_model": true,
    "text_model": true,
    "classifier_model": true,
    "encoder": true,
    "device": "cpu"
}
```

### 4. 加载模型

**POST** `/load_models`

手动加载或重新加载模型文件。

**请求体:**
```json
{
    "unsupervised_model_path": "logs/pdg/2025-05-20_14-30-00/best_pdg.pt",
    "classifier_model_path": "best_log_classifier.pt",
    "onehot_encoder_path": "onehot_encoder.pkl",
    "onehot_features_path": "onehot_feature_names.npy",
    "text_model_path": "./models/paraphrase-multilingual-MiniLM-L12-v2"
}
```

**响应示例:**
```json
{
    "status": "success",
    "message": "Models loaded successfully",
    "device": "cpu"
}
```

### 5. 单个样本预测

**POST** `/static_predict`

对单个样本进行预测。

**请求体:**
```json
{
    "component": "TestComponent",
    "case_id": "TEST_001",
    "test_suite": "SuiteA",
    "rule": "Rule1",
    "code_str": "def test_function():\n    return 'Hello World'",
    "Desc": "Test function description",
    "Func": "test_function",
    "case_spce": "Test specification",
    "case_purpose": "Test purpose"
}
```

**响应示例:**
```json
{
    "pred_label": 1,
    "prob_0": 0.23,
    "prob_1": 0.77,
    "confidence": 0.77,
    "status": "success"
}
```

### 6. 批量预测

**POST** `/static_predict_batch`

对多个样本进行批量预测。

**请求体:**
```json
{
    "batch": [
        {
            "component": "CompA",
            "case_id": "CASE_001",
            "test_suite": "Suite1",
            "rule": "RuleA",
            "code_str": "print('Hello 1')",
            "Desc": "Description 1",
            "Func": "func1",
            "case_spce": "Spec 1",
            "case_purpose": "Purpose 1"
        },
        {
            "component": "CompB",
            "case_id": "CASE_002",
            "test_suite": "Suite2",
            "rule": "RuleB",
            "code_str": "print('Hello 2')",
            "Desc": "Description 2",
            "Func": "func2",
            "case_spce": "Spec 2",
            "case_purpose": "Purpose 2"
        }
    ]
}
```

**响应示例:**
```json
{
    "results": [
        {
            "pred_label": 1,
            "prob_0": 0.15,
            "prob_1": 0.85,
            "confidence": 0.85,
            "status": "success"
        },
        {
            "pred_label": 0,
            "prob_0": 0.92,
            "prob_1": 0.08,
            "confidence": 0.92,
            "status": "success"
        }
    ],
    "count": 2,
    "status": "success"
}
```

## 数据字段说明

### 必需字段

- `code_str`: 代码字符串，用于图神经网络编码
- `component`: 组件名称，用于one-hot编码
- `case_id`: 用例ID，用于one-hot编码
- `test_suite`: 测试套件，用于one-hot编码
- `rule`: 规则名称，用于one-hot编码

### 文本字段

- `Desc`: 描述信息
- `Func`: 函数名称
- `case_spce`: 用例规格
- `case_purpose`: 用例目的

## 错误处理

FastAPI提供了更好的错误处理机制：

**错误响应示例:**
```json
{
    "detail": "模型加载失败: 文件不存在"
}
```

HTTP状态码说明：
- `200`: 请求成功
- `422`: 请求参数验证失败（FastAPI自动验证）
- `404`: 端点不存在
- `500`: 服务器内部错误

## FastAPI特性

### 类型安全
所有请求和响应都通过Pydantic模型进行类型验证，确保数据完整性。

### 自动文档
FastAPI自动生成OpenAPI文档，可通过 `/docs` 访问Swagger UI，通过 `/redoc` 访问ReDoc。

### 高性能
基于ASGI和Starlette，提供比传统WSGI框架（如Flask）更高的性能。

### 异步支持
所有端点都支持异步处理，可以处理大量并发请求。

## 测试脚本

使用提供的测试脚本验证API功能：

```bash
python test_api.py
```

或指定自定义服务器地址：

```bash
python test_api.py http://your-server:8000
```

测试脚本包含以下功能：
- 根路径测试
- API文档可用性检查
- 健康检查
- 模型状态检查
- 单样本预测
- 批量预测
- 模型加载功能

## 日志

服务器日志保存在 `api_server.log` 文件中，包含详细的运行信息和错误记录。

## 性能优化建议

1. **批量处理**: 使用 `/static_predict_batch` 接口进行批量预测，减少网络开销
2. **模型缓存**: 模型在启动时加载，避免重复加载
3. **GPU支持**: 如果可用，自动使用NPU/GPU加速推理
4. **异步处理**: 考虑使用异步框架处理大量并发请求

## 安全考虑

1. **输入验证**: 所有输入数据都会进行验证和清理
2. **错误处理**: 详细的错误信息记录在服务器端，客户端只收到必要的错误提示
3. **CORS支持**: 默认启用CORS支持，可根据需要调整

## 扩展功能

可以根据需要添加以下功能：

1. **模型版本管理**: 支持多版本模型切换
2. **A/B测试**: 支持不同模型的对比测试
3. **性能监控**: 添加推理延迟和吞吐量监控
4. **认证授权**: 添加API密钥或JWT认证
5. **限流保护**: 防止API滥用和DDoS攻击