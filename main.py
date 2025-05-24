from data_loader import load_symptoms
from faiss_utils import build_or_load_index
from gui import launch_gui

def main():
    symptoms = load_symptoms()
    vectors, index, symptom_map = build_or_load_index(symptoms)
    launch_gui(symptoms, index, symptom_map)

if __name__ == "__main__":
    main()
