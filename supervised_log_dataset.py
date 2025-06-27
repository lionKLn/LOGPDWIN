import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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
        onehot_columns = ["api_ut", "oracle_name", "component", "component_set", "module"]
        self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.onehot_features = self.onehot_encoder.fit_transform(self.df[onehot_columns])

        # 保存tags文本
        self.tags = self.df["tags"].fillna("").tolist()
        self.max_length = max_length

    # def _preprocess(self):
    #     samples = []
    #     for _, row in self.data.iterrows():
    #         # 构造输入文本
    #         text = f"[{row['api_ut']}] calls [{row['oracle_name']}] in [{row['component']}] / [{row['module']}] with tags: {row['tags']}"
    #
    #         # 标签转换
    #         label = 1 if str(row['false_positives']).strip().upper() == "TRUE" else 0
    #         print(label)
    #
    #         # 编码文本
    #         encoding = self.tokenizer(
    #             text,
    #             padding="max_length",
    #             truncation=True,
    #             max_length=self.max_length,
    #             return_tensors="pt"
    #         )
    #         print(text)
    #         sample = {
    #             "input_ids": encoding["input_ids"].squeeze(0),
    #             "attention_mask": encoding["attention_mask"].squeeze(0),
    #             "label": torch.tensor(label, dtype=torch.long)
    #         }
    #         samples.append(sample)
    #     return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # One-hot编码特征
        onehot_feat = torch.tensor(self.onehot_features[idx], dtype=torch.float)

        # CodeBERT embedding
        tag_text = self.tags[idx]
        encoded_input = self.tokenizer(tag_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            outputs = self.codebert(**{k: v for k, v in encoded_input.items()})
            # 使用CLS token表示
            tag_embedding = outputs.last_hidden_state[:,0,:].squeeze(0)  # shape: (768,)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "onehot": onehot_feat,
            "tag_embedding": tag_embedding,
            "label": label
        }



if __name__ == "__main__":
    dataset = LogDataset("dataset/data1.csv")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        print("One-hot shape:", batch["onehot"].shape)  # (B, N_onehot)
        print("Tag embedding shape:", batch["tag_embedding"].shape)  # (B, 768)
        print("Label:", batch["label"])
        break


