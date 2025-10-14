unsupervised_train/
│
├── __init__.py
├── dataset.py          # 负责从xlsx提取code_str并构建图数据
├── data_loader.py      # 从 .pt 文件加载 DataLoader
├── preprocess.py       # 负责将code_str转成CPG
├── model.py            # 定义GINEncoder和无监督训练模型
├── train.py            # 无监督训练入口（InfoGraph式训练）

Build on your machine from scratch: You need to install Miniconda and the corresponding CUDA Version 10.0. (This implementation has been tested using Python 3.6.5 and tensorflow 1.14)

2. Linux 终端用wget下载
登录 Linux 主机，进入你想保存镜像的目录（如 /home/yourname/docker_images，自定义）：
bash
# 创建目录（若不存在）
mkdir -p /home/yourname/docker_images
# 进入目录
cd /home/yourname/docker_images
执行wget命令，粘贴刚才复制的直接下载 URL：
bash
wget -c "https://zenodo.org/record/7533280/files/tailor_image.tar?download=1" -O tailor_image.tar
命令说明：
-c：支持断点续传（若下载中断，重新执行命令可继续下载，无需从头开始）。
-O tailor_image.tar：指定下载后的文件名（确保最终是 tailor_image.tar，方便后续加载）。
若 URL 包含特殊字符（如 ?），需用双引号 " 包裹 URL。