from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import google.generativeai as genai

app = Flask(__name__)

# Model paths
MODEL_PATH = 'diabetes_model.pkl'
SCALER_PATH = 'diabetes_scaler.pkl'

# Configure Gemini API
GEMINI_API_KEY = 'AIzaSyCmyNaxh1mBw12JBrQkpNF8GNIY3ns2KZE'
genai.configure(api_key=GEMINI_API_KEY)

def train_model_if_needed():
    """Train the model if it doesn't exist"""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Model already exists. Loading...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    
    print("Training new model...")
    if not os.path.exists('diabetes.csv'):
        raise FileNotFoundError("Dataset file 'diabetes.csv' not found")
    
    df = pd.read_csv('diabetes.csv')
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return model, scaler

# Load or train model at startup
model, scaler = train_model_if_needed()

def get_gemini_response(prediction, probability, top_features, input_data):
    """Generate a prompt and get a response from the Gemini API"""
    try:
        # Construct the prompt
        result_text = "Diabetic" if prediction == 1 else "Non-Diabetic"
        feature_text = ", ".join([f"{f['feature']}: {f['importance']:.4f}" for f in top_features])
        input_text = ", ".join([f"{name}: {value}" for name, value in zip(
            ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
            input_data[0]
        )])
        
        prompt = f"""
                    A machine learning model predicted that a patient is {result_text} with a confidence of {probability:.2%}.
                    The top contributing factors to this prediction are: {feature_text}.
                    The patient's input data is: {input_text}.
                    
                    Provide medical advice based on this prediction in a bullet point format, including:
                    • Risk assessment
                    • Lifestyle recommendations
                    • Monitoring suggestions
                    • When to seek medical attention
                    
                    Keep the total response under 150 words. Emphasize that this is not a definitive diagnosis and they should consult a healthcare professional.
                    """
        
        # Initialize Gemini model (assuming Gemini 1.5 Flash for efficiency)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Call Gemini API
        response = gemini_model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        return f"Error retrieving advice from Gemini API: {str(e)}"

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
        
        # Get Gemini response
        gemini_advice = get_gemini_response(prediction, probability[prediction], top_features, input_data)
        
        # Create result object
        result = {
            'prediction': int(prediction),
            'probability': float(probability[prediction]),
            'top_features': top_features,
            'gemini_advice': gemini_advice
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)