from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model, scaler, and prediction statements
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('prediction_statements.pkl', 'rb') as f:
    prediction_statements = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Age': float(request.form['age']),
            'Gender': int(request.form['gender']),
            'ChestPainType': int(request.form['chest_pain']),
            'RestingBP': float(request.form['resting_bp']),
            'Cholesterol': float(request.form['cholesterol']),
            'FastingBS': int(request.form['fasting_bs']),
            'RestECG': int(request.form['rest_ecg']),
            'MaxHR': float(request.form['max_hr']),
            'ExerciseAngina': int(request.form['exercise_angina']),
            'Oldpeak': float(request.form['oldpeak']),
            'Slope': int(request.form['slope']),
            'CA': int(request.form['ca']),
            'Thal': int(request.form['thal'])
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Scale the features
        scaled_features = scaler.transform(df)

        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]

        # Get prediction statement
        statement = prediction_statements[prediction]

        return jsonify({
            'prediction': int(prediction),
            'statement': statement,
            'probability': float(probability[prediction])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 