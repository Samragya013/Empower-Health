import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define feature names (must match training data order)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Get user input
print("Enter health data:")
preg = float(input("Pregnancies: "))
gluc = float(input("Glucose: "))
bp = float(input("Blood Pressure: "))
skin = float(input("Skin Thickness: "))
ins = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

# Create a DataFrame with feature names
input_data = pd.DataFrame([[preg, gluc, bp, skin, ins, bmi, dpf, age]], columns=feature_names)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict
risk = model.predict_proba(input_scaled)[0][1]
print(f"Diabetes Risk Score: {risk:.2%}")
if risk > 0.5:
    recommendations = "High risk: Consult a doctor."
    if bmi > 30:
        recommendations += " Focus on exercise to reduce BMI."
    if gluc > 140:
        recommendations += " Reduce sugar intake."
else:
    recommendations = "Low risk: Maintain healthy lifestyle."
    if bmi > 25:
        recommendations += " Consider light exercise."
print(recommendations)

# After calculating risk
plt.bar(['Risk Level'], [risk * 100], color='green' if risk <= 0.5 else 'red')
plt.ylim(0, 100)
plt.title('Diabetes Risk Assessment')
plt.ylabel('Risk (%)')
plt.show()