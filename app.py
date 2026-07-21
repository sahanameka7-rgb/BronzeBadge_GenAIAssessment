import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Load model and scaler
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Employee Productivity Prediction App")

st.write("Enter employee details to predict productivity score.")

# -----------------------------
# User Inputs
# -----------------------------
exp_years = st.number_input("Experience (years)", min_value=0.0, step=0.5)
training_hours = st.number_input("Training Hours", min_value=0, step=1)
working_hours = st.number_input("Working Hours", min_value=0, step=1)
projects = st.number_input("Projects Completed", min_value=0, step=1)

# -----------------------------
# Convert years → hours
# -----------------------------
exp_hours = exp_years * 8760

# Prepare input
user_data = np.array([[training_hours, working_hours, projects, exp_hours]])

# Scale input
user_scaled = scaler.transform(user_data)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Productivity Score"):
    prediction = model.predict(user_scaled)[0]
    st.success(f"Predicted Productivity Score: {prediction:.2f}")
