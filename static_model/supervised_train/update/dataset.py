import os
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import torch_npu
    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False


class FeatureDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_device():
    """
    获取当前设备。
    按你的环境要求，优先固定使用 npu:5。
    """
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device("npu:5")
    except Exception:
        pass
    return torch.device("cpu")


def set_npu_device():
    """
    显式设置当前 NPU 设备为 5 号卡。
    如果当前环境不支持 NPU，则静默回退到 CPU。
    """
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.set_device(5)
            print("当前已显式设置 NPU 设备为 npu:5")
        else:
            print("当前环境未检测到可用 NPU，使用 CPU")
    except Exception as e:
        print(f"设置 NPU 设备失败，回退到 CPU。原因: {e}")


def calculate_class_weights(y_train: torch.Tensor) -> torch.Tensor:
    """
    计算类别权重（不再显式 .to(device)）
    """
    class_count = torch.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (2 * class_count)
    return class_weights.float()


def load_encoded_pkl(pkl_path: str) -> pd.DataFrame:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"数据文件不存在: {pkl_path}")

    df = pd.read_pickle(pkl_path)

    required_cols = ["merged_features", "false_positive"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{pkl_path} 缺少必要字段: {missing}")

    return df


def df_to_tensors(df: pd.DataFrame):
    X = torch.tensor(df["merged_features"].tolist(), dtype=torch.float32)
    y = torch.tensor(df["false_positive"].tolist(), dtype=torch.long)
    return X, y