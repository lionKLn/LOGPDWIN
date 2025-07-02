import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 确保导入NPU的相关库
import torch_npu


class LogDataset(Dataset):
    def __init__(self, csv_path, codebert_model="./codebert", max_length=128):
        # 读取csv
        self.df = pd.read_csv(csv_path)

        # 标签
        self.labels = self.df["false_positives"].apply(lambda x: 1 if x == "TRUE" else 0).values

        # 初始化CodeBERT tokenizer和模型
        self.tokenizer = RobertaTokenizer.from_pretrained(codebert_model)
        self.codebert = RobertaModel.from_pretrained(codebert_model)
        self.codebert.eval()  # 不训练embedding

        # 需要做one-hot编码的列
        onehot_columns = ["api_ut", "oracle_name", "sut.component", "sut.component_set", "sut.module"]
        self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.onehot_features = self.onehot_encoder.fit_transform(self.df[onehot_columns])

        # 保存tags文本
        self.tags = self.df["tags"].fillna("").tolist()
        self.max_length = max_length

        # 将模型移动到NPU
        self.device = torch.device("npu")  # 指定使用NPU设备
        self.codebert = self.codebert.to(self.device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # One-hot编码特征
        onehot_feat = torch.tensor(self.onehot_features[idx], dtype=torch.float).to(self.device)

        # CodeBERT embedding
        tag_text = self.tags[idx]
        encoded_input = self.tokenizer(tag_text, truncation=True, padding="max_length", max_length=self.max_length,
                                       return_tensors="pt")
        # 将输入移至NPU
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            outputs = self.codebert(**encoded_input)
            # 使用CLS token表示
            tag_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  # shape: (768,)

        label = torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)

        return {
            "onehot": onehot_feat,
            "tag_embedding": tag_embedding,
            "label": label
        }


if __name__ == "__main__":
    dataset = LogDataset("dataset/data1.csv")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 在每个batch上移动数据到NPU
    for batch in dataloader:
        # 确保数据都在NPU设备上
        onehot = batch["onehot"]
        tag_embedding = batch["tag_embedding"]
        label = batch["label"]

        print("One-hot shape:", onehot.shape)  # (B, N_onehot)
        print("Tag embedding shape:", tag_embedding.shape)  # (B, 768)
        print("Label:", label)
        break
