import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# 定义分类器（保持你提供的简单单层结构）
class LogClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 二分类：0（无缺陷）/1（有缺陷）
        )

    def forward(self, x):
        return self.net(x)


# 定义数据集类
class DefectDataset(Dataset):
    def __init__(self, data_path):
        """
        加载整合后的特征数据
        data_path: 最终处理好的Excel文件路径（final_processed_data.xlsx）
        """
        self.df = pd.read_excel(data_path)
        self.labels = self.df["false_positive"].values  # 标签
        self.features = self._prepare_features()  # 特征矩阵

    def _prepare_features(self):
        """将各类特征拼接为统一的特征向量"""
        features_list = []

        # 1. 处理code_str编码（图向量）
        code_embeddings = np.array([np.array(emb) for emb in self.df["code_str_embedding"]])
        features_list.append(code_embeddings)

        # 2. 处理文本编码（Sentence-BERT向量）
        text_cols = ["Desc_embedding", "Func_embedding", "case_space_embedding", "case_purpose_embedding"]
        for col in text_cols:
            embeddings = np.array([np.array(emb) for emb in self.df[col]])
            features_list.append(embeddings)

        # 3. 处理One-hot特征（直接取数值列）
        onehot_cols = [col for col in self.df.columns if
                       col.startswith(("component_", "case_id_", "test_suite_", "rule_"))]
        onehot_features = self.df[onehot_cols].values
        features_list.append(onehot_features)

        # 拼接所有特征（按样本维度拼接）
        return np.concatenate(features_list, axis=1).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# 定义Lightning模块（封装训练逻辑）
class DefectPredictor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = LogClassifier(input_dim, hidden_dim)
        self.loss_fn = nn.CrossEntropyLoss()  # 二分类交叉熵损失
        self.val_metrics = []  # 保存验证集指标

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        labels = batch["label"]
        outputs = self(features)
        loss = self.loss_fn(outputs, labels)

        # 计算训练集准确率
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        labels = batch["label"]
        outputs = self(features)
        loss = self.loss_fn(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.cpu(), preds.cpu(), average="binary"
        )

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

        self.val_metrics.append({
            "labels": labels.cpu(),
            "preds": preds.cpu()
        })
        return loss

    def on_validation_epoch_end(self):
        """每个验证 epoch 结束后计算混淆矩阵"""
        all_labels = torch.cat([m["labels"] for m in self.val_metrics]).numpy()
        all_preds = torch.cat([m["preds"] for m in self.val_metrics]).numpy()

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["无缺陷", "有缺陷"],
                    yticklabels=["无缺陷", "有缺陷"])
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title(f"验证集混淆矩阵 (Epoch {self.current_epoch})")

        # 保存混淆矩阵图片
        Path("classifier_results/confusion_matrices").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"classifier_results/confusion_matrices/epoch_{self.current_epoch}.png")
        plt.close()

        self.val_metrics.clear()  # 清空缓存

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    # 配置
    DATA_PATH = "final_processed_data.xlsx"  # 整合后的特征数据
    BATCH_SIZE = 32
    HIDDEN_DIM = 128
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 50
    VAL_SPLIT = 0.2  # 训练集:验证集 = 8:2
    device = torch.device("npu:6" if torch.npu.is_available() else
                          "cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建输出目录
    Path("classifier_results/checkpoints").mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    dataset = DefectDataset(DATA_PATH)
    input_dim = dataset.features.shape[1]  # 自动计算输入特征维度
    print(f"✅ 数据加载完成，样本数: {len(dataset)}, 特征维度: {input_dim}")

    # 划分训练集和验证集
    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 2. 初始化模型
    model = DefectPredictor(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        lr=LEARNING_RATE
    )

    # 3. 配置训练器
    checkpoint_callback = ModelCheckpoint(
        dirpath="classifier_results/checkpoints",
        filename="best-model",
        monitor="val_f1",  # 以F1分数作为最优模型指标（比准确率更适合不平衡数据）
        mode="max",
        save_top_k=1
    )

    early_stopping = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=5,  # 5个epoch没提升就停止
        verbose=True
    )

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="npu" if "npu" in str(device) else "gpu" if "cuda" in str(device) else "cpu",
        devices=[int(str(device).split(":")[-1])] if "npu" in str(device) or "cuda" in str(device) else "auto",
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir="classifier_results",
        log_every_n_steps=10
    )

    # 4. 开始训练
    print(f"🚀 开始训练分类器，设备: {device}")
    trainer.fit(model, train_loader, val_loader)

    # 5. 输出最佳模型信息
    print(f"🎯 训练完成！最佳模型保存在: {checkpoint_callback.best_model_path}")
    print(f"最佳验证集F1分数: {checkpoint_callback.best_score:.4f}")


if __name__ == "__main__":
    main()
