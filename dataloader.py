import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel
import numpy as np

class LogDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path).fillna("")

        # Ensure labels are binary (convert to int)
        self.labels = self.df['false_positives'].astype(int).values

        # Fields
        self.onehot_fields = ['oracle_name', 'sut.component', 'sut.component_set', 'sut.module']
        self.codebert_fields = ['api_ut', 'tags']

        # One-Hot Encoding
        self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.onehot_encoded = self.onehot_encoder.fit_transform(self.df[self.onehot_fields])

        # Load CodeBERT
        self.tokenizer = AutoTokenizer.from_pretrained("./codebert")
        self.model = AutoModel.from_pretrained("./codebert")
        self.model.eval()

        if hasattr(torch, 'npu'):
            self.device = torch.device("npu" if torch.npu.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Encode CodeBERT fields
        self.apt_ut_embeddings = self.encode_column(self.df['api_ut'])
        self.tag_embeddings = self.encode_column(self.df['tags'])

        self.onehot_dim = self.onehot_encoded.shape[1]
        self.codebert_dim = 768  # CodeBERT 默认输出维度

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        onehot = torch.tensor(self.onehot_encoded[idx], dtype=torch.float)
        apt_ut_embed = torch.tensor(self.apt_ut_embeddings[idx], dtype=torch.float)
        tag_embed = torch.tensor(self.tag_embeddings[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        features = torch.cat([onehot, apt_ut_embed, tag_embed], dim=-1)
        return features, label

    def encode_column(self, column):
        embeddings = []
        with torch.no_grad():
            for text in column:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
                embeddings.append(cls_embedding.squeeze(0).cpu().numpy())
        return embeddings

    def get_feature_dim(self):
        return self.onehot_dim + 2 * self.codebert_dim

def get_dataloader(csv_path, batch_size=16, shuffle=True):
    dataset = LogDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    dataloader = get_dataloader("your_data.csv", batch_size=8)
    dataset = LogDataset("your_data.csv")
    print("总特征维度：", dataset.get_feature_dim())
    for features, labels in dataloader:
        print("Features shape:", features.shape)  # [B, D]
        print("Labels shape:", labels.shape)  # [B]