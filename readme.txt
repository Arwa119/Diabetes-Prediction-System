# Diabetes Prediction Web Application

A beautiful, user-friendly web application for diabetes prediction using machine learning. This application takes patient information as input and predicts whether the patient is diabetic or non-diabetic with a confidence score.

![Diabetes Prediction App Screenshot](https://via.placeholder.com/800x400?text=Diabetes+Prediction+App)

## Features

- **Beautiful UI/UX Design:** Modern, responsive interface with intuitive input forms
- **Machine Learning Backend:** Random Forest classifier for accurate predictions
- **Real-time Prediction:** Instant feedback after submitting patient information
- **Confidence Score:** Provides probability of the prediction
- **Feature Importance:** Shows top factors contributing to the prediction
- **Medical Advice:** Basic guidance based on prediction results

## Technologies Used

- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Flask (Python)
- **Machine Learning:** scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model:** Random Forest Classifier

## Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone the repository or download files

Save the provided files to your project directory.

### Step 2: Create the project structure

```
diabetes-prediction/
├── app.py               # Main Flask application
├── ne.csv               # Your diabetes dataset
├── templates/
│   └── index.html       # HTML template
├── static/              # (Optional) For static files
└── requirements.txt     # Dependencies
```

### Step 3: Install dependencies

Create a `requirements.txt` file with the following content:

```
flask==2.2.3
pandas==1.5.3
numpy==1.24.2
scikit-learn==1.2.2
joblib==1.2.0
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Create the templates directory

```bash
mkdir templates
```

### Step 5: Move HTML template to templates directory

Copy the content of the HTML template provided and save it as `templates/index.html`.

### Step 6: Run the application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000/` in your web browser.

## How to Use

1. Open the application in your web browser
2. Enter patient information in the form:
   - Number of pregnancies
   - Glucose level
   - Blood pressure
   - Skin thickness
   - Insulin level
   - BMI (Body Mass Index)
   - Diabetes pedigree function
   - Age
3. Click "Predict" to get the result
4. View the prediction (Diabetic or Non-Diabetic) with confidence score
5. Review the top contributing factors for the prediction
6. Read the basic medical advice provided

## Model Details

The application uses a Random Forest classifier, which is an ensemble learning method for classification. The model is trained on the provided diabetes dataset with the following features:

- **Pregnancies:** Number of times pregnant
- **Glucose:** Plasma glucose concentration
- **BloodPressure:** Diastolic blood pressure (mm Hg)
- **SkinThickness:** Triceps skin fold thickness (mm)
- **Insulin:** 2-Hour serum insulin (mu U/ml)
- **BMI:** Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction:** Diabetes pedigree function
- **Age:** Age in years

The model produces a binary classification:
- **0:** Non-Diabetic
- **1:** Diabetic

## Important Notes

- The model is only as good as the data it's trained on
- This application is for educational purposes only
- Always consult healthcare professionals for proper medical advice
- Regular model retraining is recommended as new data becomes available

## Customization

You can customize the application by:
- Modifying the HTML/CSS for a different look and feel
- Adjusting the model parameters for potentially better accuracy
- Adding more features to the prediction form
- Implementing user authentication for secure access
- Adding data visualization components for better insights

## License

This project is for educational purposes only. Use at your own discretion.

---

Developed with ❤️ for diabetes prediction and awareness.