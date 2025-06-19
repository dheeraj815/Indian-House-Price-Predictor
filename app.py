# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# Load model, scaler, and encoder
model = joblib.load(os.path.join(MODEL_DIR, "indian_house_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(MODEL_DIR, "location_encoder.pkl"))

st.set_page_config(
    page_title="Indian House Price Predictor", layout="centered")
st.title("🏠 Indian House Price Predictor (Bangalore)")

# Locations from encoder
location_labels = list(encoder.classes_)


def get_user_input():
    location = st.selectbox("Location", sorted(location_labels))
    total_sqft = st.slider("Total Square Feet", 500, 5000, 1000)
    bath = st.slider("Number of Bathrooms", 1, 5, 2)
    bhk = st.slider("Number of Bedrooms (BHK)", 1, 5, 2)

    # Encode location
    location_encoded = encoder.transform([location])[0]

    input_data = pd.DataFrame([{
        'location': location_encoded,
        'total_sqft': total_sqft,
        'bath': bath,
        'bhk': bhk
    }])

    return input_data


input_df = get_user_input()

if st.button("Predict Price 💰"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏷️ Estimated House Price: ₹ {prediction:.2f} Lakhs")
