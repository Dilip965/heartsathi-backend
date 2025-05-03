import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Define paths to the model, scaler, and statements files
MODEL_PATH = os.getenv('MODEL_PATH', './rf_model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', './scaler.pkl')
STATEMENTS_PATH = os.getenv('STATEMENTS_PATH', './prediction_statements.pkl')

# Load model, scaler, and statements
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(STATEMENTS_PATH, 'rb') as f:
        prediction_statements = pickle.load(f)
    logging.info("Loaded model, scaler, and prediction statements.")
except Exception as e:
    logging.error(f"Error loading artifacts: {e}")
    raise

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"message": "Heart Disease Prediction API is up"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        required_fields = [
            'Age', 'Gender', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'Slope', 'CA', 'Thal'
        ]

        # Validate payload
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # Prepare features
        df = pd.DataFrame([data])
        X = scaler.transform(df)

        # Predict
        pred_class = int(model.predict(X)[0])
        pred_proba = float(model.predict_proba(X)[0][pred_class])
        statement = prediction_statements[pred_class]

        return jsonify({
            'prediction': pred_class,
            'probability': pred_proba,
            'statement': statement
        }), 200

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Bind to 0.0.0.0 and use PORT env var (default 5000)
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
