supervised_train/
│
├── codebert/           #负责图数据编码
├── model/ #负责编码
    ├── paraphrase-multilingual-MiniLM-L12-v2/ #负责编码文本数据
├── pdg_model/          # 预训练好的代码编码模型
├── infer.py       # 推理新数据的代码
├── model.py      # 分类使用的模型
├── dataset.py          # 从xlsx下面加载数据并编码
├── preprocess.py       # 负责将code_str转成CPG
├── train_supervised.py            # 监督训练入口

具体的运行步骤为：
前置步骤为通过无监督训练得到了代码编码模型
首先修改好dataset.py下的文件路径和保存路径
然后运行python dataset.py生成processed_dataset.pkl（训练需要的编码后的数据）

之后运行python train_supervised.py训练模型

推理过程
运行python infer.py来推理新数据，将添加新数据的标签，标签为0的概率，标签为1的概率，后续可根据概率进行排序

