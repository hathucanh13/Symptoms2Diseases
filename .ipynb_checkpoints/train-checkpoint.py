import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

file_pf = "archive/dataset_Diseases_and_Symptoms.csv"
df = pd.read_csv(file_pf)


#Data processing:
all_symptoms = df.columns[1:].tolist()# Extract all values from the first column
unique_first_field_values = df.iloc[:, 0].drop_duplicates()
# Transpose
unique_first_field_values_df = pd.DataFrame(unique_first_field_values).T


tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().numpy()

