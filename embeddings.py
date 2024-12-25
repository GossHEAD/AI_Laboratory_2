import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=50)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
    return embeddings
