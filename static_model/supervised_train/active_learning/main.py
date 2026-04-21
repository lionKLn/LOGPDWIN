from data_loader import load_full_data, split_labeled_pool
from active_loop import active_learning_loop

PKL_PATH = "processed_dataset.pkl"


def main():
    # 1️⃣ 加载数据
    X, y = load_full_data(PKL_PATH)

    # 2️⃣ 初始划分
    X_labeled, X_pool, y_labeled, y_pool = split_labeled_pool(
        X, y, init_ratio=0.1
    )

    print(f"初始 labeled: {len(X_labeled)}, pool: {len(X_pool)}")

    # 3️⃣ 主动学习
    model,history = active_learning_loop(
        X_labeled,
        X_pool,
        y_labeled,
        y_pool,
        rounds=10,
        query_size=200,
        sampling_strategy="uncertainty_sampling",
        mode_save_path="final_active_model.pt"
    )

    print(history)

    print("\n主动学习完成！")


if __name__ == "__main__":
    main()