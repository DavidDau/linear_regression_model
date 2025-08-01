"""
Cardiovascular Disease Prediction Model Training and Saving Script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def load_and_prepare_data():
    """Load and prepare the cardiovascular dataset"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, '..', 'linear_regression', 'cardio_base.csv')
        
        # Load the real dataset
        df = pd.read_csv(csv_path)
        if df is None or df.empty:
            raise FileNotFoundError(f"CSV file is empty or could not be loaded: {csv_path}")
            
        print(f"Successfully loaded data from: {csv_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns found: {df.columns.tolist()}")

        # Add missing columns with default values
        required_columns = ['id', 'age', 'gender', 'height', 'weight', 
                          'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 
                          'smoke', 'alco', 'active', 'cardio']
                          
        for col in required_columns:
            if col not in df.columns:
                if col == 'gluc':
                    df[col] = 1  # Normal glucose level
                elif col in ['smoke', 'alco']:
                    df[col] = 0  # Non-smoker, non-drinker
                elif col == 'active':
                    df[col] = 1  # Active lifestyle
                elif col == 'cardio':
                    # Calculate cardio based on other risk factors
                    df[col] = ((df['ap_hi'] > 140) | 
                             (df['ap_lo'] > 90) | 
                             (df['cholesterol'] > 1)).astype(int)
        
        # Basic data cleaning
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Remove any rows with missing values
        df = df.dropna()
        
        # Feature engineering
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        df['age_risk'] = (df['age'] > 50).astype(int)
        df['bp_risk'] = ((df['ap_hi'] > 140) | (df['ap_lo'] > 90)).astype(int)
        df['lifestyle_risk'] = df['smoke'] + df.get('alco', 0) - df.get('active', 1)
        
        # Drop redundant features
        features_to_drop = ['height', 'weight']
        X = df.drop(['cardio'] + features_to_drop, axis=1)
        y = df['cardio']
        
        print("Data preparation complete")
        print(f"Final feature set: {X.columns.tolist()}")
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        raise

def train_models():
    """Train multiple models and select the best one"""
    X, y = load_and_prepare_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'test_mse': test_mse,
            'test_r2': test_r2
        }
        
        print(f"{name} - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.6f}")
    
    # Select best model based on R² score
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
    
    return best_model, scaler, X.columns.tolist()

def save_model():
    """Train and save the best model"""
    print("Training cardiovascular disease prediction models...")
    
    best_model, scaler, feature_names = train_models()
    
    # Get the directory where this script is located
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save model, scaler, and feature names in the script directory
    with open(os.path.join(script_dir, 'cardio_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(os.path.join(script_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(script_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("\n Model saved as 'cardio_model.pkl'")
    print(" Scaler saved as 'scaler.pkl'")
    print(" Feature names saved as 'feature_names.pkl'")
    
    return feature_names

def load_model():
    """Load the trained model, scaler, and feature names"""
    try:
        # Get the directory where this script is located
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Script directory: {script_dir}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Look for model files in the script directory
        model_files = {
            'cardio_model.pkl': os.path.join(script_dir, 'cardio_model.pkl'),
            'scaler.pkl': os.path.join(script_dir, 'scaler.pkl'),
            'feature_names.pkl': os.path.join(script_dir, 'feature_names.pkl')
        }
        
        # Check if all files exist
        missing_files = [name for name, path in model_files.items() if not os.path.exists(path)]
        
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        with open(model_files['cardio_model.pkl'], 'rb') as f:
            model = pickle.load(f)
        
        with open(model_files['scaler.pkl'], 'rb') as f:
            scaler = pickle.load(f)
        
        with open(model_files['feature_names.pkl'], 'rb') as f:
            feature_names = pickle.load(f)
        
        return model, scaler, feature_names
    except Exception as e:
        # Instead of retraining, raise an error with instructions
        raise Exception(f"Model files not found or corrupted: {e}. Please run 'python prediction.py' to train the model first.")

def predict_cardiovascular_disease(input_data):
    """
    Make prediction for cardiovascular disease
    
    Args:
        input_data (dict): Dictionary containing patient data
        
    Returns:
        dict: Prediction result with probability and risk level
    """
    model, scaler, feature_names = load_model()
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Add missing columns with defaults if they don't exist
    if 'gluc' not in df.columns:
        df['gluc'] = input_data.get('gluc', 1)  # Default to normal glucose
    if 'alco' not in df.columns:
        df['alco'] = input_data.get('alco', 0)  # Default to no alcohol
    if 'active' not in df.columns:
        df['active'] = input_data.get('active', 1)  # Default to active
    
    # Feature engineering (same as training)
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['age_risk'] = (df['age'] > 50).astype(int)
    df['bp_risk'] = ((df['ap_hi'] > 140) | (df['ap_lo'] > 90)).astype(int)
    df['lifestyle_risk'] = df['smoke'] + df['alco'] - df['active']
    
    # Drop height and weight (replaced by BMI)
    df = df.drop(['height', 'weight'], axis=1)
    
    # Ensure features are in the correct order - only use features that exist
    # If some features are missing from the training data, fill with defaults
    missing_features = [col for col in feature_names if col not in df.columns]
    for feature in missing_features:
        if feature == 'gluc':
            df[feature] = 1
        elif feature == 'alco':
            df[feature] = 0
        elif feature == 'active':
            df[feature] = 1
        else:
            df[feature] = 0  # Default for any other missing features
    
    # Now select features in the correct order
    df = df[feature_names]
    
    # Standardize features
    df_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    
    # Convert to probability (clamp between 0 and 1)
    probability = max(0, min(1, prediction))
    
    # Determine risk level
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    return {
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "has_disease": probability > 0.5
    }

if __name__ == "__main__":
    # Train and save the model
    save_model()
    
    # Test prediction
    test_data = {
        "age": 45,
        "gender": 1,
        "height": 170,
        "weight": 75,
        "ap_hi": 130,
        "ap_lo": 85,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }
    
    result = predict_cardiovascular_disease(test_data)
    print(f"\nTest prediction: {result}")
