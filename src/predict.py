import os
import joblib
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoders.pkl')

def predict(patient_data):
    # 1. Load Model and Encoders
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Run src/train.py first.")
        return

    print("Loading model and encoders...")
    model = joblib.load(MODEL_PATH)
    # encoders = joblib.load(ENCODER_PATH) # Load this if inputs are strings like 'Male'

    # 2. Prepare Data
    # Your input was already numeric, so we convert it to a 2D array directly.
    # If your input was a dictionary of strings, we would use the 'encoders' here.
    patient_array = np.array([patient_data])

    # 3. Predict
    print(f"Analyzing patient data: {patient_data}")
    prediction = model.predict(patient_array)
    probability = model.predict_proba(patient_array)

    # 4. Output
    print("-" * 30)
    if prediction[0] == 1:
        print(f"RESULT: Heart Disease Detected (Confidence: {probability[0][1]:.2f})")
    else:
        print(f"RESULT: No Heart Disease (Confidence: {probability[0][0]:.2f})")
    print("-" * 30)

if __name__ == "__main__":
    # Example input from your original code
    # [Age, Sex, ChestPain, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, etc...]
    new_patient = [45, 1, 2, 120, 230, 0, 1, 150, 0, 1.5, 1, 0, 2]
    
    predict(new_patient)