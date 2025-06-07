from tkinter import *
from utils.faiss_utils import find_similar
from utils.data_loader import load_symptoms_from_dataset
from utils.model_utils import get_s2d_model
from config import BG_IMAGE_PATH 

def launch_gui(symptoms, index, symptom_map):
    root = Tk()
    root.title("Symptoms to Disease Prediction")
    root.geometry("720x480")
    root.configure(bg="lightblue")
    bg = PhotoImage(file= BG_IMAGE_PATH) 
    bg_label = Label(root, image=bg)
    bg_label.place(y=0,x=1,relwidth=1, relheight=1)

    Label(root, text="Hello! Type your symptoms below! :)", bg="lightblue", font=("Helvetica")).pack(pady=20)

    input_frame = Frame(root, bg="lightblue")
    input_frame.pack(side="bottom", fill="x")

    input_field = Entry(input_frame, width=75, font=("Helvetica"), bg="white")
    input_field.pack(padx=0, pady=5, side="left")
    user_symptoms = []
    def reset():
        user_symptoms.clear()
        input_field.config(state="normal")
        input_field.delete(0, END)
        for widget in root.pack_slaves()[2:]:  # Clear old results
            widget.destroy()
    def predict():
        input_field.config(state="disabled")
        for widget in root.pack_slaves()[2:]:  # Clear old results
            widget.destroy()

        if not user_symptoms:
            Label(root, text="Please select at least one symptom.", bg="lightblue").pack()
            input_field.config(state="normal")
            return
        all_symptoms = load_symptoms_from_dataset()
        user_symptom_vector = [1 if symptom in user_symptoms else 0 for symptom in all_symptoms]       
        rf = get_s2d_model()
        probs = rf.predict_proba([user_symptom_vector])[0]
        top_indices = probs.argsort()[-3:][::-1]
        top_diagnoses = [(rf.classes_[i], probs[i]) for i in top_indices]
        print("Top diagnoses:", top_diagnoses)
        print ("User symptoms:", user_symptoms)
        Label(root, text=f"Selected symptoms: {', '.join(user_symptoms)}", bg="lightblue").pack()

        Label(root, text="Predicted diagnoses:", bg="lightblue").pack()
        for diagnosis, prob in top_diagnoses:
            Label(root, text=f"{diagnosis} (probability: {prob * 100:.4f})", bg="lightblue").pack()

        Button(root, text="Start Over", bg="darkblue", font=("Helvetica"), fg="white", command=reset).pack(pady=10)
        Label(root, text="DISCLAIMER: This is not a diagnosis. Please visit a medical facility for a full check-up", bg="lightblue", font=("Helvetica")).pack()        

    def analyze():
        user_input = input_field.get()
        input_field.delete(0, END)
        input_field.config(state="disabled")
        for widget in root.pack_slaves()[2:]:  # Clear old results
            widget.destroy()
        results = find_similar(index, symptom_map, user_input)
        Label(root, text="Great! I understand it as follow, tick if the symptoms  apply to you: ", bg="lightblue").pack()
        selected_vars, selected_syms = [], []

        for symptom, score in results:
            var = IntVar()
            chk = Checkbutton(root, text=f"{symptom} (distance: {score:.4f})", bg="lightblue",
                              font=("Helvetica"), variable=var)
            chk.pack()
            selected_vars.append(var)
            selected_syms.append(symptom)
        def on_yes():
            for widget in root.pack_slaves()[2:]:
                widget.destroy()
            input_field.config(state="normal")
            Label(root, text="Type your symptoms below!", bg="lightblue", font=("Helvetica")).pack(pady=20)

        def ask():
            Label(root, text="Do you want to provide further symptoms?", bg="lightblue").pack()
            Button(root, text="Yes", bg="darkblue", font=("Helvetica"), fg="white", command=on_yes).pack(pady=5)
            Button(root, text="No, start predicting", bg="darkblue", font=("Helvetica"), fg="white", command=predict).pack(pady=5)
        def on_ok():
            checked = [sym for var, sym in zip(selected_vars, selected_syms) if var.get()]
            for symptom in checked:
                if symptom not in user_symptoms:
                    user_symptoms.append(symptom)
            for widget in root.pack_slaves()[2:]:
                widget.destroy()
            ask()
        Button(root, text="OK", bg="darkblue", font=("Helvetica"), fg="white", command=on_ok).pack(padx=5)

    Button(input_frame, text="➤", bg="darkblue", font=("Helvetica"), fg="white", command=analyze).pack(side="left", padx=5)
    root.mainloop()
