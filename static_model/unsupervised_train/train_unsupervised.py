import os
import torch
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

# 导入NPU相关依赖（若未安装需先执行：pip install torch-npu）
try:
    import torch_npu
except ImportError:
    raise ImportError("请先安装torch-npu以支持NPU设备，安装命令：pip install torch-npu")

from unsupervised_train.model import GAE_GIN_lightning
from unsupervised_train.dataloader import GraphDataset


def main():
    # 1. 基础配置
    data_dir = "../data/graph_dataset"  # 图数据目录（.pt文件）
    batch_size = 8  # 批大小
    num_workers = 4  # 数据加载进程数
    target_npu = "npu:6"  # 目标NPU卡号（可根据需求修改）
    max_epochs = 20  # 训练轮次
    checkpoint_dir = "./checkpoints"  # 权重保存目录

    # 2. 设备检测与配置
    # 检查NPU是否可用，优先使用指定的NPU，否则降级到GPU/CPU
    npu_available = torch.npu.is_available() and target_npu in [f"npu:{i}" for i in range(torch.npu.device_count())]
    if npu_available:
        device = torch.device(target_npu)
        accelerator = "npu"
        devices = [int(target_npu.split(":")[-1])]  # Lightning需要传入卡号列表
        print(f"✅ 检测到可用NPU设备：{target_npu}，将使用该设备训练")
    else:
        # 若NPU不可用，降级到GPU/CPU
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            accelerator = "gpu"
            devices = [0]
            print(f"⚠️ 指定的NPU设备{target_npu}不可用，降级使用GPU:0")
        else:
            device = torch.device("cpu")
            accelerator = "cpu"
            devices = "auto"
            print(f"⚠️ NPU和GPU均不可用，使用CPU训练（训练速度可能较慢）")

    # 设置默认设备（确保数据和模型都加载到指定设备）
    torch.npu.set_device(device) if npu_available else None

    # 3. 数据加载
    # 加载图数据集，自动读取目录下所有.pt文件
    train_dataset = GraphDataset(data_dir)
    # 构建DataLoader，批量加载数据（shuffle=True打乱训练数据）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # 开启内存锁定，加速数据传输到NPU/GPU
    )
    print(f"✅ 成功加载数据集：共{len(train_dataset)}个图样本，批大小={batch_size}")

    # 4. 模型初始化（关键：in_channels需与data.x维度一致）
    # 若data.x是CodeBERT编码（768维），需将in_channels改为768；若为其他维度则对应修改
    model = GAE_GIN_lightning(
        in_channels=768,  # 重点：必须与data.x的维度一致（原代码128需根据实际数据修改）
        out_channels=768,  # 保留接口兼容性，无实际作用
        batch_size=batch_size,
        encoder_kwargs={
            "num_gc_layers": 2,  # GIN层数（可调整）
            "hidden_dim": 128  # GIN每层输出维度（可调整）
        }
    )
    # 将模型移动到指定设备
    model.to(device)
    print(f"✅ 模型初始化完成，已加载到{device}设备")

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",  # 监控指标
        save_top_k=1,  # 保存最优的一个模型
        mode="min",  # 越小越好
        dirpath=os.path.join(checkpoint_dir, "best"),  # 保存目录
        filename="gin-unsupervised-best",  # 文件名
    )

    # 5. 训练器配置（适配NPU）
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,  # 设备类型：npu/gpu/cpu
        devices=devices,  # 具体设备号：NPU/GPU卡号列表，CPU设为auto
        log_every_n_steps=10,  # 每10步记录一次日志
        default_root_dir=checkpoint_dir,  # 权重和日志保存根目录
        enable_checkpointing=True,  # 开启权重保存
        callbacks=[checkpoint_callback],  # ✅ 通过callbacks列表传入
    )

    # 6. 启动训练
    print("🚀 开始无监督训练...")
    print(f"📊 训练配置：设备={device}，轮次={max_epochs}，批大小={batch_size}")
    trainer.fit(model, train_loader)
    print("🎯 训练完成！最优权重已保存到：", os.path.join(checkpoint_dir, "best"))


if __name__ == "__main__":
    main()