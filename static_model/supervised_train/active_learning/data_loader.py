import pandas as pd
import torch


def load_full_data(pkl_path):
    df = pd.read_pickle(pkl_path)

    X = torch.tensor(df["merged_features"].tolist(), dtype=torch.float32)
    y = torch.tensor(df["false_positive"].tolist(), dtype=torch.long)

    return X, y


def split_labeled_pool(X, y, init_ratio=0.1, seed=42):
    from sklearn.model_selection import train_test_split

    X_labeled, X_pool, y_labeled, y_pool = train_test_split(
        X, y,
        test_size=1 - init_ratio,
        random_state=seed,
        stratify=y
    )

    return X_labeled, X_pool, y_labeled, y_pool