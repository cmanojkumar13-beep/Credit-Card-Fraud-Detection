import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/best_fraud_model.pkl")

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")

# Select one transaction
sample = X_test.iloc[[0]]

# Predict
prediction = model.predict(sample)[0]

probability = model.predict_proba(sample)[0][1]

print("Prediction:", prediction)
print("Fraud Probability:", probability)

if prediction == 1:
    print("Result: Fraudulent Transaction")
else:
    print("Result: Genuine Transaction")