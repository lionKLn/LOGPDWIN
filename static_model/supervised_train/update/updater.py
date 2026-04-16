import os
import copy
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model import LogClassifier
from dataset import (
    FeatureDataset,
    get_device,
    set_npu_device,
    calculate_class_weights,
    load_encoded_pkl,
    df_to_tensors,
)


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value, model):
        score = metric_value

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(model)
            self.counter = 0

    def _save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"验证指标提升，模型已保存到 {self.path}")


def compute_metrics(y_true, y_pred, pos_label=0) -> Dict:
    return {
        "overall": {
            "accuracy": accuracy_score(y_true, y_pred)
        },
        f"class_{pos_label}": {
            "precision": precision_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0),
        },
        f"class_{1 - pos_label}": {
            "precision": precision_score(y_true, y_pred, average="binary", pos_label=1 - pos_label, zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", pos_label=1 - pos_label, zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", pos_label=1 - pos_label, zero_division=0),
        },
    }


def merge_labeled_data(
    old_labeled_pkl: str,
    new_labeled_pkl: str,
    save_merged_pkl: Optional[str] = None,
) -> pd.DataFrame:
    old_df = load_encoded_pkl(old_labeled_pkl)
    new_df = load_encoded_pkl(new_labeled_pkl)

    merged_df = pd.concat([old_df, new_df], axis=0, ignore_index=True)

    # 如果后续你有唯一标识字段，可以在这里去重
    # 例如：
    # merged_df = merged_df.drop_duplicates(subset=["sample_id"], keep="last")

    if save_merged_pkl:
        merged_df.to_pickle(save_merged_pkl)
        print(f"合并后的 labeled 数据已保存到: {save_merged_pkl}")

    return merged_df


def build_model(
    input_dim: int,
    hidden_dim: int,
    base_model_path: Optional[str] = None,
) -> LogClassifier:
    """
    构建模型并尽可能加载旧模型参数。
    不再显式调用 .to(device)。
    """
    model = LogClassifier(input_dim=input_dim, hidden_dim=hidden_dim)

    if base_model_path and os.path.exists(base_model_path):
        try:
            state_dict = torch.load(base_model_path)
            model.load_state_dict(state_dict)
            print(f"已加载旧模型参数: {base_model_path}")
        except Exception as e:
            raise RuntimeError(f"加载旧模型失败: {e}")
    else:
        print("未提供旧模型或旧模型不存在，将从头初始化模型。")

    return model


def train_updated_model(
    model: LogClassifier,
    X: torch.Tensor,
    y: torch.Tensor,
    save_model_path: str,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 5e-4,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    pos_label: int = 0,
    use_early_stopping: bool = False,
) -> Tuple[LogClassifier, Dict]:
    """
    在合并后的 labeled 数据上更新模型。
    按你的环境要求，不在训练过程中使用 .to(device)。
    """
    # 显式设置 NPU:5
    set_npu_device()
    current_device = get_device()
    print(f"当前训练设备: {current_device}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=y,
    )

    train_dataset = FeatureDataset(X_train, y_train)
    val_dataset = FeatureDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    class_weights = calculate_class_weights(y_train)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    stopper = None
    if use_early_stopping:
        stopper = EarlyStopping(
            patience=5,
            delta=0.001,
            verbose=True,
            path=save_model_path,
        )

    best_val_f1 = -1.0
    best_state = None

    print("\n开始模型更新训练...")
    for epoch in range(1, epochs + 1):
        # ===== Train =====
        model.train()
        train_loss = 0.0
        train_preds, train_true = [], []

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            # 按你的要求，不再做 batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
            train_true.extend(batch_y.detach().cpu().numpy())

        train_avg_loss = train_loss / len(train_loader.dataset)
        train_metrics = compute_metrics(train_true, train_preds, pos_label=pos_label)

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_avg_loss = val_loss / len(val_loader.dataset)
        val_metrics = compute_metrics(val_true, val_preds, pos_label=pos_label)
        val_f1 = val_metrics[f"class_{pos_label}"]["f1"]

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_avg_loss:.4f} | "
            f"Train Acc: {train_metrics['overall']['accuracy']:.4f}, "
            f"Train F1({pos_label}): {train_metrics[f'class_{pos_label}']['f1']:.4f} | "
            f"Val Loss: {val_avg_loss:.4f} | "
            f"Val Acc: {val_metrics['overall']['accuracy']:.4f}, "
            f"Val F1({pos_label}): {val_f1:.4f}"
        )

        scheduler.step()

        if use_early_stopping:
            stopper(val_f1, model)
            if stopper.early_stop:
                print("早停触发，训练结束。")
                break
        else:
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())

    # ===== 保存最佳模型 =====
    if use_early_stopping:
        best_model = LogClassifier(
            input_dim=X.shape[1],
            hidden_dim=model.net[0].out_features
        )
        best_model.load_state_dict(torch.load(save_model_path))
    else:
        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, save_model_path)

        best_model = LogClassifier(
            input_dim=X.shape[1],
            hidden_dim=model.net[0].out_features
        )
        best_model.load_state_dict(best_state)

    print(f"\n更新后的模型已保存到: {save_model_path}")

    # ===== 最终验证集评估 =====
    best_model.eval()
    final_preds, final_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = best_model(batch_x)
            final_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            final_true.extend(batch_y.numpy())

    final_metrics = compute_metrics(final_true, final_preds, pos_label=pos_label)

    print("\n==================== 更新后模型验证集性能 ====================")
    print(f"整体准确率：{final_metrics['overall']['accuracy']:.4f}")
    print(f"{pos_label}类 Precision：{final_metrics[f'class_{pos_label}']['precision']:.4f}")
    print(f"{pos_label}类 Recall：{final_metrics[f'class_{pos_label}']['recall']:.4f}")
    print(f"{pos_label}类 F1：{final_metrics[f'class_{pos_label}']['f1']:.4f}")
    print("===========================================================")

    return best_model, final_metrics


def update_model_with_new_labeled_data(
    old_labeled_pkl: str,
    new_labeled_pkl: str,
    base_model_path: Optional[str],
    updated_model_save_path: str,
    merged_labeled_save_path: str,
    hidden_dim: int = 128,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 5e-4,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    pos_label: int = 0,
    use_early_stopping: bool = False,
):
    """
    工程入口：
    1. 合并历史 labeled 与新增 labeled
    2. 加载旧模型
    3. 基于合并后的 labeled 数据更新模型
    """
    # 显式设置 NPU:5
    set_npu_device()
    current_device = get_device()
    print(f"当前运行设备: {current_device}")

    # 1. 合并新旧 labeled 数据
    merged_df = merge_labeled_data(
        old_labeled_pkl=old_labeled_pkl,
        new_labeled_pkl=new_labeled_pkl,
        save_merged_pkl=merged_labeled_save_path,
    )

    # 2. 转张量
    X, y = df_to_tensors(merged_df)

    # 3. 构建模型并加载旧参数
    model = build_model(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        base_model_path=base_model_path,
    )

    # 4. 更新模型
    updated_model, metrics = train_updated_model(
        model=model,
        X=X,
        y=y,
        save_model_path=updated_model_save_path,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        random_seed=random_seed,
        pos_label=pos_label,
        use_early_stopping=use_early_stopping,
    )

    return updated_model, metrics