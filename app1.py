from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model, scaler, and prediction statements (using environment variables for production)
MODEL_PATH = os.getenv('MODEL_PATH', 'rf_model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'scaler.pkl')
PREDICTION_STATEMENTS_PATH = os.getenv('PREDICTION_STATEMENTS_PATH', 'prediction_statements.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    with open(PREDICTION_STATEMENTS_PATH, 'rb') as f:
        prediction_statements = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading model/scaler/prediction statements: {e}")
    raise

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Heart Disease Prediction API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # List of required fields
        required_fields = [
            'Age', 'Gender', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'Slope', 'CA', 'Thal'
        ]

        # Check if all required fields are present in the incoming data
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing one or more required fields.'}), 400

        # Create DataFrame for the incoming data
        df = pd.DataFrame([data])

        # Scale the features using the scaler
        scaled_features = scaler.transform(df)

        # Predict using the model
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][prediction]
        statement = prediction_statements[prediction]

        # Return the prediction, probability, and statement
        return jsonify({
            'prediction': int(prediction),
            'statement': statement,
            'probability': float(probability)
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Set to False to disable debug mode in production
    app.run(host='0.0.0.0', port=5000, debug=False)
