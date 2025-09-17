# Huawei 动态日志推理脚本

本项目提供了一个基于 **PyTorch + CodeBERT + OneHotEncoder** 的推理脚本 `infer.py`，可对华为动态日志进行二分类预测，并输出预测标签及对应概率。

## 目录结构
```
.
├── infer.py # 推理脚本
├── encoder.pkl # 训练时保存的 OneHotEncoder 对象
├── encoder_columns.npy # OneHot 编码列顺序
├── log_classifier.pt # 训练好的分类器模型权重
├── codebert/ # CodeBERT 模型目录（需提前下载）
└── new_data.csv # 待推理的日志数据（CSV 格式）
```

## 环境要求

⚠️ **重要**：推理环境必须和训练环境版本保持一致，否则可能出现编码维度不一致、API 不兼容等错误。  
建议直接使用训练时导出的 `requirements.txt` 安装环境：
```bash
pip install -r requirements.txt

python==3.10

torch==2.2.0
transformers==4.26.1
scikit-learn==1.2.1
pandas==1.5.2
numpy==1.24.2
joblib==1.4.2
```

## 数据格式要求
推理数据文件需为 CSV 格式，包含以下字段：
oracle_name
sut.component
sut.component_set
sut.module
api_ut 
tags 

## 推理步骤
### 准备模型文件
确保以下文件在项目根目录：
encoder.pkl（训练时的 OneHotEncoder 对象）
encoder_columns.npy（训练时的 OneHot 列顺序）
log_classifier.pt（训练好的模型权重）
codebert/（训练时使用的 CodeBERT 模型目录）

### 运行推理
python infer.py

默认推理 new_data.csv 并生成 inference_results.csv。

### 推理结果
inference_results.csv 会在原数据基础上增加三列：
predicted_label：预测类别（0 或 1）
prob_class_0：预测为类别 0 的概率
prob_class_1：预测为类别 1 的概率

## 其他
如果想更换推理数据集，直接修改 infer.py 中的：
csv_path = "new_data.csv"
推理使用 CPU，如需使用 GPU，可将：
device = torch.device("cpu")
改为：
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 服务
pip install fastapi uvicorn
启动
uvicorn service:app --host 0.0.0.0 --port 8000 --reload