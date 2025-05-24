import pandas as pd
from config import DATASET

def load_symptoms():
    df = pd.read_csv(DATASET)
    df = df.columns[1:]
    return df
