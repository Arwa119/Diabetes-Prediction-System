import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Check if model already exists
MODEL_PATH = 'diabetes_model.pkl'
SCALER_PATH = 'diabetes_scaler.pkl'

def load_and_prepare_data():
    """Load the diabetes dataset and prepare it for training"""
    try:
        # Check if file exists in current directory
        if os.path.exists('diabetes.csv'):
            df = pd.read_csv('diabetes.csv')
        else:
            print("Error: Dataset file 'ne.csv' not found in current directory.")
            sys.exit(1)
            
        # Split into features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)
    
    except Exception as e:
        print(f"Error loading or processing the dataset: {e}")
        sys.exit(1)

def train_model():
    """Train a Random Forest model on the diabetes dataset"""
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare_data()
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel trained successfully.")
    print(f"Model accuracy: {accuracy:.4f}\n")
    
    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance:")
    for idx, row in importance_df.iterrows():
        print(f"{row['Feature']:25} {row['Importance']:.4f}")
    
    # Save model and scaler
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Scaler saved to {SCALER_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    return model, scaler

def get_user_input():
    """Get input values from user via command line"""
    print("\n=== Diabetes Prediction System ===")
    print("Please enter the following medical information:")
    
    try:
        pregnancies = float(input("Number of Pregnancies: "))
        glucose = float(input("Glucose Level (mg/dL): "))
        blood_pressure = float(input("Blood Pressure (mm Hg): "))
        skin_thickness = float(input("Skin Thickness (mm): "))
        insulin = float(input("Insulin Level (mu U/ml): "))
        bmi = float(input("BMI: "))
        diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
        age = float(input("Age: "))
        
        # Create input array
        user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                insulin, bmi, diabetes_pedigree, age]])
        
        return user_input
    
    except ValueError:
        print("Error: Please enter valid numerical values.")
        return None
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)

def make_prediction(model, scaler):
    """Make diabetes prediction based on user input"""
    user_input = get_user_input()
    
    if user_input is None:
        return
    
    # Scale the input
    user_input_scaled = scaler.transform(user_input)
    
    # Make prediction
    prediction = model.predict(user_input_scaled)
    probability = model.predict_proba(user_input_scaled)
    
    # Display result
    print("\n=== Prediction Result ===")
    if prediction[0] == 1:
        print("Result: POSITIVE - Patient is classified as Diabetic")
        print(f"Confidence: {probability[0][1]:.2%}")
    else:
        print("Result: NEGATIVE - Patient is classified as Non-Diabetic")
        print(f"Confidence: {probability[0][0]:.2%}")
    
    print("\nNote: This is a prediction only. Please consult with a healthcare professional for proper diagnosis.")

def main():
    """Main function to run the diabetes prediction system"""
    print("=== Diabetes Prediction System ===")
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            print(f"Loading existing model from {MODEL_PATH}...")
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            model, scaler = train_model()
    else:
        print("No existing model found. Training new model...")
        model, scaler = train_model()
    
    # Prediction loop
    while True:
        make_prediction(model, scaler)
        
        again = input("\nDo you want to make another prediction? (y/n): ")
        if again.lower() != 'y':
            break

if __name__ == "__main__":
    main()