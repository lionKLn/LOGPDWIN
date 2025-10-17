import sys
sys.path.append('../..')
import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 优先导入NPU支持
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except ImportError:
    npu_available = False

from model import GAE_GIN
from static_model.unsupervised_train.dataset import GraphDataset
from static_model.unsupervised_train.data_loader import CPGDataLoader


class EarlyStopping:
    """手动实现EarlyStopping逻辑"""
    def __init__(self, patience=10, verbose=False, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存当前最优模型"""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Trainer_Wrapper:
    def __init__(self, graph_type, npu_device_id=0):
        self.graph_type = graph_type
        self.npu_device_id = npu_device_id  # NPU设备ID，默认为0

        # === 设备设置（优先NPU，其次GPU，最后CPU） ===
        if npu_available:
            self.device = torch.device(f"npu:{npu_device_id}")
            torch.npu.set_device(self.device)  # 设置当前使用的NPU设备
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # === 日志设置 ===
        run_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        self.log_dir = f'logs/{graph_type}/{run_name}'
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # === 模型初始化 ===
        self.model = GAE_GIN(768, 768).to(self.device)

        # === 优化器 ===
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)

        # === 数据加载 ===
        dataset = GraphDataset(projects=['nginx'])
        train_data = dataset[: int(0.7 * len(dataset))]
        test_data = dataset[int(0.7 * len(dataset)) :]

        self.train_loader = CPGDataLoader(
            train_data, batch_size=512, graph_type=graph_type, shuffle=True, num_workers=8
        )
        self.test_loader = CPGDataLoader(
            test_data, batch_size=512, graph_type=graph_type, shuffle=False, num_workers=8
        )

        # === 提前停止 ===
        self.early_stopping = EarlyStopping(
            patience=10,
            verbose=True,
            path=os.path.join(self.log_dir, f"best_{graph_type}.pt")
        )

        print(f"Trainer initialized on {self.device} with log_dir={self.log_dir}")

    def train_one_epoch(self, epoch):
        """单个epoch的训练逻辑"""
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch in loop:
            self.optimizer.zero_grad()
            # 将数据移至指定设备（支持NPU/GPU/CPU）
            batch = batch[self.graph_type].to(self.device)
            loss = self.model.training_step(batch, batch_idx=0)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        return avg_loss

    def validate(self, epoch):
        """验证逻辑"""
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                # 将数据移至指定设备（支持NPU/GPU/CPU）
                batch = batch[self.graph_type].to(self.device)
                val_loss = self.model.validation_step(batch, batch_idx=0)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(self.test_loader)
        self.writer.add_scalar("Loss/val", avg_val_loss, epoch)
        return avg_val_loss

    def train(self, max_epochs=200):
        print("Start training ...")
        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        print("Training finished.")
        self.writer.close()


# 使用示例
if __name__ == "__main__":
    # 可指定使用的NPU设备ID（默认为0）
    trainer = Trainer_Wrapper(graph_type='pdg', npu_device_id=4)
    trainer.train(max_epochs=200)
