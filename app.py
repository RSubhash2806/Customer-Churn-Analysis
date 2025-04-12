import streamlit as st
import joblib
import numpy as np

# Load the scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# App Title
st.title("Churn Prediction App")
st.divider()

st.write("Please enter the values and tap the predict button to get a prediction.")
st.divider()

# Input fields
age = st.number_input("Enter your age:", min_value=8, max_value=100, value=30)
gender = st.selectbox("Select Gender:", ["Male", "Female"])
tenure = st.number_input("Enter tenure (months):", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("Enter Monthly Charge:", min_value=30, max_value=150, value=50)

st.divider()

# Predict Button
predict_button = st.button("PREDICT")

if predict_button:
    # Convert gender to binary (example encoding)
    gender_encoded = 1 if gender == "Female" else 0

    # Construct feature list in correct order
    features = [age, gender_encoded, tenure, monthlycharge]

    # Convert to array and reshape
    input_array = np.array([features])

    try:
        # Scale the input
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Interpret prediction
        predicted_label = "Yes" if prediction == 1 else "No"

        st.success(f"Churn Predicted: {predicted_label}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Please fill in all the values and click on 'PREDICT'.")
