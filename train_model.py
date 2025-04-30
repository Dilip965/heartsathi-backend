import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('Heart4.csv')

# Separate features and target
X = df.drop('Output', axis=1)
y = df['Output']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Calculate accuracy
train_accuracy = rf_model.score(X_train_scaled, y_train)
test_accuracy = rf_model.score(X_test_scaled, y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Save the model and scaler
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define prediction statements
prediction_statements = {
    0: "No heart disease detected. Your heart health appears to be good. Continue maintaining a healthy lifestyle.",
    1: "Mild heart disease detected. Consider consulting a healthcare provider for preventive measures.",
    2: "Moderate heart disease detected. Please schedule an appointment with a cardiologist for further evaluation.",
    3: "Severe heart disease detected. Urgent medical attention required. Please visit a healthcare facility immediately.",
    4: "Critical heart disease detected. Emergency medical attention required. Please seek immediate medical care."
}

# Save prediction statements
with open('prediction_statements.pkl', 'wb') as f:
    pickle.dump(prediction_statements, f) 