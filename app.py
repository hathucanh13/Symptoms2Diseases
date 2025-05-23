import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tkinter import *
import os
import pickle
file_pf = "archive/all_symptoms.csv"
df = pd.read_csv(file_pf)


#Data processing:
all_symptoms = df.iloc[:, 0].values.tolist()
embedModel = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

def get_embedding(text):
    return embedModel.encode(text)

if os.path.exists("symptom_vectors.npy") and os.path.exists("symptom_index.faiss") and os.path.exists("symptom_map.pkl"):
    print("Loading precomputed FAISS index and vectors...")
    symptom_vectors = np.load("symptom_vectors.npy ")
    index = faiss.read_index("symptom_index.faiss")
    with open("symptom_map.pkl", "rb") as f:
        symptom_map = pickle.load(f)
else:
    print("Computing symptom embeddings and building FAISS index...")
    symptom_vectors = np.array([embedModel.encode(s) for s in all_symptoms]).astype('float32')
    index = faiss.IndexFlatL2(symptom_vectors.shape[1])
    index.add(symptom_vectors)
    symptom_map = {i: all_symptoms[i] for i in range(len(all_symptoms))}
    
    # Save everything for future use
    np.save("symptom_vectors.npy", symptom_vectors)
    faiss.write_index(index, "symptom_index.faiss")
    with open("symptom_map.pkl", "wb") as f:
        pickle.dump(symptom_map, f)

# Store using FAISS
# dim = symptom_vectors.shape[1]
# index.add(symptom_vectors)

# np.save("symptom_vectors.npy", symptom_vectors)
# faiss.write_index(index, "symptom_index.faiss")

# Map vector index to original symptom
def find_similar_symptoms(user_input, top_k=5):
    query_vec = get_embedding(user_input).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    return [(symptom_map[i], distances[0][j]) for j, i in enumerate(indices[0])]

# import pickle
# with open("symptom_map.pkl", "wb") as f:
#     pickle.dump(symptom_map, f)
root = Tk()
root.title("Symptoms to Disease Prediction")
root.configure(bg="lightblue")
root.geometry("480x480")

my_label = Label(root, text ="Hello! Type your symptoms below! :) ", bg="lightblue", font=("Helvetica"), fg="black") 
my_label.pack(pady=20)

# Add a frame at the bottom
bottom_frame = Frame(root, bg="lightblue")
bottom_frame.pack(side="bottom", fill="x", pady=20)

input_field = Entry(bottom_frame, width=47, bg ="white", font=("Helvetica"), fg="black")
input_field.pack(side="left", padx=5)
def onClick():
    # Get the input from the input field
    user_input = input_field.get()
    # Print the input to the console
    input_field.delete(0, END)
    # Call the function to find similar symptoms
    results = find_similar_symptoms(user_input)
    # Print the results
    print("Matching symptoms:")
    for symptom, score in results:
        print(f"- {symptom} (distance: {score:.4f})")

send_button = Button(bottom_frame, text ="➤", bg="darkblue", font=("Helvetica"), fg="white", command=onClick)
send_button.pack(side="left", padx=5)


root.mainloop()
