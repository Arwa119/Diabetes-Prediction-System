from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Model paths
MODEL_PATH = 'diabetes_model.pkl'
SCALER_PATH = 'diabetes_scaler.pkl'

def train_model_if_needed():
    """Train the model if it doesn't exist"""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Model already exists. Loading...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    
    print("Training new model...")
    # Load data
    if not os.path.exists('diabetes.csv'):
        raise FileNotFoundError("Dataset file 'diabetes.csv' not found")
    
    df = pd.read_csv('diabetes.csv')
    
    # Split into features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Get feature names for later use
    feature_names = list(X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return model, scaler

# Load or train model at startup
model, scaler = train_model_if_needed()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['bloodPressure'])
        skin_thickness = float(request.form['skinThickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetesPedigree'])
        age = float(request.form['age'])
        
        # Create input array
        input_data = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]])
        
        # Scale the data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Get feature importances
        feature_importance = model.feature_importances_
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Create list of feature importances
        importances = []
        for name, importance in zip(feature_names, feature_importance):
            importances.append({
                'feature': name,
                'importance': float(importance)
            })
        
        # Sort by importance
        importances = sorted(importances, key=lambda x: x['importance'], reverse=True)
        
        # Get top 3 features
        top_features = importances[:3]
        
        # Create result object
        result = {
            'prediction': int(prediction),
            'probability': float(probability[prediction]),
            'top_features': top_features
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)