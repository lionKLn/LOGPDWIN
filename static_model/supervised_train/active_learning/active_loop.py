import torch
from trainer import train_model_active, predict_proba
from sampler import uncertainty_sampling


from sklearn.metrics import f1_score, accuracy_score
import torch



#显式使用npu5
torch.npu.set_device(5)

def evaluate(model, X_test, y_test, device, pos_label=0):
    model.eval()
    preds = []

    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    y_true = y_test.numpy()

    return {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, pos_label=pos_label)
    }


def active_learning_loop(
    X_labeled,
    X_pool,
    y_labeled,
    y_pool,
    X_test,
    y_test,
    rounds=10,
    query_size=100,
    sampling_strategy="uncertainty_sampling"
):
    #device = get_device()

    history = []

    for r in range(rounds):
        print(f"\n===== Round {r} =====")

        # 1️⃣ 划分 validation（从 labeled 中切）

        # 2️⃣ 训练模型
        model = train_model_active(
            X_train=X_labeled, y_train=y_labeled,
        )

        # 3️⃣ 测试集评估（🔥关键）
        metrics = evaluate(model, X_test, y_test, device)

        print(f"Test Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

        history.append(metrics)

        # 4️⃣ 主动学习采样
        probs = predict_proba(model, X_pool)
        selected_idx = uncertainty_sampling(probs, query_size)

        # 更新数据
        X_selected = X_pool[selected_idx]
        y_selected = y_pool[selected_idx]

        X_labeled = torch.cat([X_labeled, X_selected])
        y_labeled = torch.cat([y_labeled, y_selected])

        mask = torch.ones(len(X_pool), dtype=torch.bool)
        mask[selected_idx] = False

        X_pool = X_pool[mask]
        y_pool = y_pool[mask]

        if len(X_pool) == 0:
            break

    # 保存最终模型
    torch.save(model.state_dict(), "final_active_model.pt")

    return model, history