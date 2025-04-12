import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction")

st.divider()

age = st.number_input("Enter your age: ", min_value=8, max_value=100, value=30)

gender = st.selectbox("Select the Gender", ["Male", "Female"])

tenure = st.number_input("Enter the value of the Tenure: ", min_value=0, max_value=130, value=10)

monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)

st.divider()

predictButton = st.button("PREDICT")

if predictButton:

    gender_selected = 1 if gender == "Female" else 0

    x = [age, gender_selected, tenure, monthlycharge]  # ✅ Corrected here

    x_array = scaler.transform([x])  # already a list of one sample

    prediction = model.predict(x_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.success(f"Predicted: {predicted}")

else:
    st.info("Please enter the values and tap the PREDICT button")
