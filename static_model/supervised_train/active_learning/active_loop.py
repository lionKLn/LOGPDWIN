import torch
from trainer import train_model, predict_proba
from sampler import uncertainty_sampling


def active_learning_loop(
    X_labeled,
    X_pool,
    y_labeled,
    y_pool,
    rounds=10,
    query_size=100
):
    for r in range(rounds):
        print(f"\n===== Active Learning Round {r} =====")

        # 1️⃣ 训练模型
        model = train_model(
            X_train=X_labeled,
            y_train=y_labeled,
            epochs=20
        )

        # 2️⃣ 预测 pool
        probs = predict_proba(model, X_pool)

        # 3️⃣ 选择最不确定样本
        selected_idx = uncertainty_sampling(probs, query_size)

        # 4️⃣ 获取样本
        X_selected = X_pool[selected_idx]
        y_selected = y_pool[selected_idx]

        # 5️⃣ 更新 labeled
        X_labeled = torch.cat([X_labeled, X_selected])
        y_labeled = torch.cat([y_labeled, y_selected])

        # 6️⃣ 更新 pool
        mask = torch.ones(len(X_pool), dtype=torch.bool)
        mask[selected_idx] = False

        X_pool = X_pool[mask]
        y_pool = y_pool[mask]

        print(f"Labeled size: {len(X_labeled)} | Pool size: {len(X_pool)}")

        if len(X_pool) == 0:
            print("Pool empty, stop.")
            break

    return model