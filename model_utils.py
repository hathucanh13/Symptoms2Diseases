from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL
import joblib

# Load the model once
embedModel = SentenceTransformer(EMBED_MODEL)                            
s2d_model = joblib.load("rf-v2.pkl")
def get_embedding(text):
    return embedModel.encode(text)

def get_model():
    return embedModel

def get_s2d_model():
    return s2d_model