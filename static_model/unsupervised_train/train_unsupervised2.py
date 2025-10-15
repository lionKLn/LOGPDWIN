import os
import time
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 用于日志记录（替代 Lightning 日志）

# 导入NPU相关依赖（若未安装需先执行：pip install torch-npu）
try:
    import torch_npu
except ImportError:
    raise ImportError("请先安装torch-npu以支持NPU设备，安装命令：pip install torch-npu")

# 导入纯PyTorch版模型（替换原 Lightning 版 GAE_GIN_lightning）
from unsupervised_train.model import GAE_GIN  # 需确保 model.py 中是修改后的纯PyTorch版
from unsupervised_train.dataloader import GraphDataset


def main():
    # 1. 基础配置（与原代码一致）
    data_dir = "../data/graph_dataset"  # 图数据目录（.pt文件）
    batch_size = 8  # 批大小
    num_workers = 4  # 数据加载进程数
    target_npu = "npu:6"  # 目标NPU卡号（可根据需求修改）
    max_epochs = 20  # 训练轮次
    checkpoint_dir = "./checkpoints"  # 权重保存目录
    log_dir = os.path.join(checkpoint_dir, "logs")  # 日志保存目录
    best_model_path = os.path.join(checkpoint_dir, "best", "gin-unsupervised-best.pth")  # 最优权重路径

    # 2. 设备检测与配置（保留原NPU优先逻辑）
    # 检查NPU是否可用，优先使用指定的NPU，否则降级到GPU/CPU
    npu_available = torch.npu.is_available() and target_npu in [f"npu:{i}" for i in range(torch.npu.device_count())]
    if npu_available:
        device = torch.device(target_npu)
        torch.npu.set_device(device)  # 设置默认NPU设备
        print(f"✅ 检测到可用NPU设备：{target_npu}，将使用该设备训练")
    else:
        # 若NPU不可用，降级到GPU/CPU
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"⚠️ 指定的NPU设备{target_npu}不可用，降级使用GPU:0")
        else:
            device = torch.device("cpu")
            print(f"⚠️ NPU和GPU均不可用，使用CPU训练（训练速度可能较慢）")

    # 3. 目录创建（确保日志和权重目录存在）
    os.makedirs(os.path.join(checkpoint_dir, "best"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # 初始化TensorBoard日志（替代 Lightning 的日志功能）
    writer = SummaryWriter(log_dir=log_dir)

    # 4. 数据加载（与原代码一致，保留pin_memory加速）
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

    # 5. 模型初始化（替换为纯PyTorch版 GAE_GIN）
    model = GAE_GIN(
        in_channels=768,  # 重点：必须与data.x的维度一致（原代码128需根据实际数据修改）
        out_channels=768,  # 保留接口兼容性，无实际作用
        device=device,  # 传入指定设备（NPU/GPU/CPU）
        encoder_kwargs={
            "num_gc_layers": 2,  # GIN层数（可调整）
            "hidden_dim": 128  # GIN每层输出维度（可调整）
        }
    )
    print(f"✅ 模型初始化完成，已加载到{device}设备")

    # 6. 优化器与训练配置（替代 Lightning 的 Trainer 和 configure_optimizers）
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # 与原Lightning学习率一致
    best_train_loss = float("inf")  # 记录最优损失（用于保存最优权重）
    log_every_n_steps = 10  # 每10步记录一次日志（与原Lightning一致）

    # 7. 手动训练循环（核心：替代 Lightning 的 trainer.fit）
    print("🚀 开始无监督训练...")
    print(f"📊 训练配置：设备={device}，轮次={max_epochs}，批大小={batch_size}")

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()
        model.train()  # 开启训练模式
        total_loss = 0.0
        step = 0

        # 单轮次训练（遍历所有批次）
        for batch_idx, batch in enumerate(train_loader):
            step += 1
            # 单批次训练（调用模型的 train_step 方法）
            batch_loss = model.train_step(batch, optimizer)
            total_loss += batch_loss

            # 按步记录日志（与原 log_every_n_steps 一致）
            if step % log_every_n_steps == 0:
                current_step = (epoch - 1) * len(train_loader) + step
                writer.add_scalar("Train/Batch Loss", batch_loss, current_step)
                print(f"Epoch [{epoch}/{max_epochs}] | Step {step}/{len(train_loader)} | Batch Loss: {batch_loss:.6f}")

        # 计算本轮平均损失
        epoch_avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        # 记录epoch级日志
        writer.add_scalar("Train/Epoch Avg Loss", epoch_avg_loss, epoch)
        print(f"\n[Epoch {epoch:03d}/{max_epochs}] "
              f"Train Avg Loss: {epoch_avg_loss:.6f} | "
              f"Time: {epoch_time:.2f}s\n")

        # 保存最优权重（替代 Lightning 的 ModelCheckpoint）
        if epoch_avg_loss < best_train_loss:
            best_train_loss = epoch_avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🔍 检测到更优模型（损失从 {best_train_loss:.6f} 降至 {epoch_avg_loss:.6f}），已保存到：{best_model_path}")

    # 8. 训练完成后清理
    writer.close()  # 关闭TensorBoard日志
    print("🎯 训练完成！")
    print(f"📁 最优权重保存路径：{best_model_path}")
    print(f"📊 训练日志保存路径：{log_dir}（可通过 tensorboard --logdir {log_dir} 查看）")


if __name__ == "__main__":
    main()