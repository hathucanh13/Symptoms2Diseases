from utils.data_loader import load_symptoms_from_dataset
from utils.faiss_utils import build_or_load_index
from gui.gui import launch_gui

def main():
    symptoms = load_symptoms_from_dataset()
    vectors, index, symptom_map = build_or_load_index(symptoms)
    launch_gui(symptoms, index, symptom_map)

if __name__ == "__main__":
    main()
