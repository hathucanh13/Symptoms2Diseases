import pandas as pd
from config import SYMPTOM_FILE

def load_symptoms_from_dataset():
    df = pd.read_csv(SYMPTOM_FILE)
    symptom_list = pd.Index(df['diseases'])
    return symptom_list
