from sklearn.model_selection import train_test_split
import torch
import pandas as pd


def load_and_split_active_learning(pkl_path, test_size=0.2, init_ratio=0.1, seed=42):
    df = pd.read_pickle(pkl_path)

    X = torch.tensor(df["merged_features"].tolist(), dtype=torch.float32)
    y = torch.tensor(df["false_positive"].tolist(), dtype=torch.long)

    # ① 固定 Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ② 从 Train 中划分 labeled / pool
    X_labeled, X_pool, y_labeled, y_pool = train_test_split(
        X_train_full,
        y_train_full,
        test_size=1 - init_ratio,
        random_state=seed,
        stratify=y_train_full
    )

    return X_labeled, X_pool, y_labeled, y_pool, X_test, y_test