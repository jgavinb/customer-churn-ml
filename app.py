import streamlit as st
import joblib
import numpy as np

scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/model.pkl')

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value = 100, value = 30)

tenure = st.number_input("Enter tenure", min_value = 0, max_value = 130, value = 10)

monthlycharge = st.number_input("Enter monthly charge", min_value = 30, max_value = 150)

gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()

predict_button = st.button("Predict")

st.divider()

if predict_button:
    gender_selected = 1 if gender == "Female" else 0
    
    X = [age, gender_selected, tenure, monthlycharge]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Concern" if prediction == 1 else "Non-concern"

    st.write(f"Predicted: {predicted}")
else:
    st.write("Please enter the values and hit the predict button for getting a prediction.")
