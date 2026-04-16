from updater import update_model_with_new_labeled_data


if __name__ == "__main__":
    model, metrics = update_model_with_new_labeled_data(
        old_labeled_pkl="historical_labeled.pkl",
        new_labeled_pkl="new_labeled_batch.pkl",
        base_model_path="best_log_classifier.pt",
        updated_model_save_path="best_log_classifier_v2.pt",
        merged_labeled_save_path="historical_labeled_v2.pkl",
        hidden_dim=128,
        batch_size=32,
        epochs=20,
        learning_rate=5e-4,
        val_ratio=0.2,
        random_seed=42,
        pos_label=0,
        use_early_stopping=False
    )

    print("\n模型更新完成。")
    print(metrics)