import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Loads CSV data."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None

def encode_features(X, fit=True, encoders=None):
    """
    Encodes categorical features.
    
    Args:
        X (pd.DataFrame): Features dataframe.
        fit (bool): If True, creates new encoders (Training mode). 
                    If False, uses existing encoders (Prediction mode).
        encoders (dict): Dictionary of fitted LabelEncoders (needed if fit=False).
    
    Returns:
        X (pd.DataFrame): Encoded dataframe.
        encoders (dict): The dictionary of encoders used.
    """
    object_cols = X.select_dtypes(include='object').columns
    
    if fit:
        encoders = {}
        for col in object_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    else:
        # Use existing encoders for new data
        if encoders is None:
            raise ValueError("Encoders must be provided when fit=False")
            
        for col, le in encoders.items():
            if col in X.columns:
                # Handle unknown labels gracefully (optional safety)
                X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                
    return X, encoders