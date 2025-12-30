Heart Disease PredictorThis is a Machine Learning application that uses Logistic Regression to predict heart disease risk based on clinical data like age, cholesterol, and heart rate.📂 Folder Structuredata/: Stores the raw CSV dataset.models/: Stores the trained AI model (.pkl).src/: Contains Python scripts for training and prediction.requirements.txt: List of necessary libraries.🚀 How to Use (All Systems)1. Setup (First Time Only)Open your terminal (PowerShell on Windows, Terminal on Mac/Linux) and run:Bash# Clone or enter your project folder
cd heart-disease-app

# Create a virtual environment
python -m venv venv

# Activate it:
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install libraries
pip install -r requirements.txt
2. Train the ModelYou must train the AI before you can make predictions. Ensure your CSV file is in data/raw/.Bashpython src/train.py
This will create a heart_model.pkl file in the models folder.3. Predict Patient RiskTo test a new patient's data, run the prediction script:Bashpython src/predict_patient.py
💻 System CompatibilityFeatureWindows (PowerShell)Mac / LinuxPath SlashesUses \Uses /Python Commandpythonpython3Activation.\venv\Scripts\activatesource venv/bin/activate
