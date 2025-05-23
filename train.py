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


tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", model_max_length=512)
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
print(all_symptoms[0:3])

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().numpy()

import faiss

symptoms = [
    "headache",
    "fever and chills",
    "muscle pain",
    "nausea and vomiting",
    "shortness of breath",
    "persistent cough",
    "rash on skin"
]

# Generate embeddings
symptom_vectors = np.array([get_embedding(s) for s in all_symptoms]).astype('float32')

# Store using FAISS
dim = symptom_vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(symptom_vectors)

# Map vector index to original symptom
symptom_map = {i: all_symptoms[i] for i in range(len(all_symptoms))}
def find_similar_symptoms(user_input, top_k=3):
    query_vec = get_embedding(user_input).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    return [(symptom_map[i], distances[0][j]) for j, i in enumerate(indices[0])]
user_input = "I have body pain and feel hot"
results = find_similar_symptoms(user_input)

print("Matching symptoms:")
for symptom, score in results:
    print(f"- {symptom} (distance: {score:.4f})")
