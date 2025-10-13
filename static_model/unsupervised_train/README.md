unsupervised_train/
│
├── __init__.py
├── dataset.py          # 负责从xlsx提取code_str并构建图数据
├── data_loader.py      # 从 .pt 文件加载 DataLoader
├── preprocess.py       # 负责将code_str转成CPG
├── model.py            # 定义GINEncoder和无监督训练模型
├── train.py            # 无监督训练入口（InfoGraph式训练）

