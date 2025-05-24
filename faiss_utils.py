import os
import pickle
import numpy as np
import faiss
from config import VEC_FILE, INDEX_FILE, MAP_FILE
from model_utils import get_embedding

def build_or_load_index(symptoms):
    if all(os.path.exists(f) for f in [VEC_FILE, INDEX_FILE, MAP_FILE]):
        vectors = np.load(VEC_FILE)
        index = faiss.read_index(INDEX_FILE)
        with open(MAP_FILE, "rb") as f:
            symptom_map = pickle.load(f)
    else:
        vectors = np.array([get_embedding(s) for s in symptoms]).astype('float32')
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        symptom_map = {i: s for i, s in enumerate(symptoms)}
        np.save(VEC_FILE, vectors)
        faiss.write_index(index, INDEX_FILE)
        with open(MAP_FILE, "wb") as f:
            pickle.dump(symptom_map, f)
    return vectors, index, symptom_map

def find_similar(index, symptom_map, user_input, top_k=5):
    query_vec = get_embedding(user_input).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    return [(symptom_map[i], distances[0][j]) for j, i in enumerate(indices[0])]
