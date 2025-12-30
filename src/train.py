import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import our custom preprocessing module
from preprocess import load_data, encode_features

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'HeartDiseaseTrain-Test.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoders.pkl')

def train():
    # 1. Load Data
    df = load_data(DATA_PATH)
    if df is None: return

    # 2. Separate Features and Target
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. Preprocessing (Encoding)
    print("Encoding categorical features...")
    X_encoded, encoders = encode_features(X, fit=True)

    # 4. Train/Test Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # 5. Model Training
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 7. Save Model and Encoders
    print("Saving artifacts...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Encoders saved to: {ENCODER_PATH}")

if __name__ == "__main__":
    train()