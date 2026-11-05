
import streamlit as st
import pandas as pd
import joblib

# Load model and test data
model = joblib.load("models/best_fraud_model.pkl")
X_test = pd.read_csv("data/processed/X_test.csv")

st.set_page_config(page_title="Credit Card Fraud Detection")

st.title("Credit Card Fraud Detection")
st.write("Predict whether a transaction is fraudulent or genuine.")

# Use one test transaction as example input
sample = X_test.iloc[0]

st.subheader("Transaction Details")

input_data = {}

with st.form("fraud_prediction_form"):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(X_test.columns):
        with col1 if i % 2 == 0 else col2:
            input_data[feature] = st.number_input(
                feature,
                value=float(sample[feature]),
                format="%.6f"
            )

    submit = st.form_submit_button("Predict Transaction")

if submit:
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Genuine Transaction")

    st.write(f"Fraud Probability: {probability:.2%}")
