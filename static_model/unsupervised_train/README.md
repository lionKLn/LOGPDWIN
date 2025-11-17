ccpg/
unsupervised_train/
│
├── __init__.py
├── dataprocess.py      # 负责从xlsx提取code_str并构建图数据
├── preprocess.py       # 负责将code_str转化为pdg图
├── data_loader.py      # 根据掩码加载相应的图
├── dataset.py          # 从 .pt 文件加载 DataLoader
├── preprocess.py       # 负责将code_str转成CPG
├── model.py            # 定义GINEncoder和无监督训练模型
├── train_unsupervised.py            # 无监督训练入口（InfoGraph式训练）

具体的运行步骤为：
首先修改好dataprocess.py下的文件路径和保存路径
然后运行python dataprocess.py生成代码图

之后运行python train_unsupervised.py训练模型

