import streamlit as t
import numpy as np
import pickle

# -----------------------------
# Load model and scaler
# -----------------------------
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("knn_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

t.title("Loan Defaulter Prediction App")

t.write("Enter candidate details to predict Risk score.")

# -----------------------------
# User Inputs
# -----------------------------
Age = t.number_input("Age", min_value=25.0, step=0.5)
AnnualIncome = t.number_input("AnnualIncome(lakhs)", min_value=0.0, step=0.5)
Creditscore = t.number_input("CreditScore(300-900)", min_value=0, step=1)
LoanAmount = t.number_input(" LoanAmount(lakhs)", min_value=0, step=1)
LoanTerm = t.number_input(" LoanTerm(years)", min_value=0.0, step=0.5, value=0.0)
EmpType = t.selectbox(
    "Employment Type",
    ["Self-Employed", "Salaried"]
)

# -----------------------------
# Convert years → hours
# -----------------------------

if EmpType=='Self_Employeed':
    EmpT = 1
else:
    EmpT =0
# Prepare input
user_data = np.array([[Age,AnnualIncome, Creditscore, LoanAmount, LoanTerm]])

# Scale input
user_scaled = scaler.transform(user_data)
user_scaled = np.append(user_scaled, [[EmpT]], axis=1)

# -----------------------------
# Predict
# -----------------------------
if t.button("Predict Risk Score"):
    prediction = model.predict(user_scaled)[0]
    t.success(f"Predicted Risk Score: {prediction:.2f} if score:0.00 non-defaulter, otherwise defaulter")
