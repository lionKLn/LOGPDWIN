import os
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "/root/models/paraphrase-multilingual-MiniLM-L12-v2"

print("MODEL_PATH =", MODEL_PATH)
print("存在吗？", os.path.exists(MODEL_PATH))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)

text = "你好，世界！"
inputs = tokenizer(text, return_tensors="pt")
embeddings = model(**inputs).last_hidden_state.mean(dim=1)

print("编码向量维度：", embeddings.shape)
