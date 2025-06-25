import streamlit as st
from PIL import Image
import numpy as np
import tempfile
from hybrid_model import predict_disease, recommend_crop
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Hybrid AI Crop System", layout="centered")
st.title("🌾 AI-Powered Crop Advisor")
st.markdown("Upload a leaf image and enter soil parameters to get disease detection and crop recommendation.")

# === Leaf Image Upload ===
image_file = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"," JPG"])
if image_file:
    img = Image.open(image_file)
    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
    
# === Load Disease-Solution CSV ===
disease_solution_df = pd.read_csv('plantvillage_disease_treatments.csv')
solution_model = joblib.load('solution_model.pkl')
solution_vectorizer = joblib.load('solution_vectorizer.pkl')
crop_model = joblib.load('crop_rf_model_rcc.pkl')
scaler = joblib.load('crop_scaler_rcc.pkl')
# === Get Solution for Disease ===
def solu_crop(disease):
    disease_row = disease_solution_df[disease_solution_df['Disease'] == disease]
    if not disease_row.empty:
        crop = disease_row['Crop'].values[0]
        disease_name_detail = disease_row['Disease_Name'].values[0]
        symptoms = disease_row['Symptoms'].values[0]
        solution = disease_row['Solution'].values[0]

        solu_data = f"Disease: {disease_name_detail}, Crop: {crop}, Symptoms: {symptoms}, Solution: {solution}"
        input_vectorized = solution_vectorizer.transform([solu_data])
        solution_prediction = solution_model.predict(input_vectorized)
        return crop, disease_name_detail, symptoms, solution, solution_prediction[0]
    else:
        return None, None, None, None, "Disease not found in solution dataset."
# === Predict Button ===
if st.button("🚀 Predict Disease"):
    if image_file:
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            img_path = tmp_file.name
            img.save(img_path, format='PNG')  # Ensure the format is specified if necessary

        # Predictions
        disease = predict_disease(img_path)
        #crop = recommend_crop([n, p, k, temp, humidity, ph, rainfall])

        # Results
        #st.success(f"🌱 Recommended Crop: **{crop}**")
        st.success(f"🌿 Detected Disease: **{disease}**")
        st.success(f"🦠 Predicted Disease: {disease}")
        st.markdown("### 🛠️ Disease Solution Info")    
        crop, dis_name, symptoms, solution_text, treatment = solu_crop(disease)
        st.write(f"🌱 Crop: {crop}")
        st.write(f"📝 Disease Detail: {dis_name}")
        st.write(f"⚠️ Symptoms: {symptoms}")
        st.write(f"💊 Suggested Treatment: {solution_text}")
        st.write(f"🤖 Predicted Treatment Category: {treatment}")
    else:
        st.error("❗ Please upload an image to proceed.")
        st.error("Please upload a leaf image.")
        st.warning("⚠️ No solution info found for the predicted disease.")
        
# === Soil Data Input ===
st.markdown("### 🌾 Recommended Crop Based on Soil")
st.subheader("🧪 Enter Soil Parameters")
n = st.number_input("Nitrogen (N)", value=90)
p = st.number_input("Phosphorus (P)", value=42)
k = st.number_input("Potassium (K)", value=43)
temp = st.number_input("Temperature (°C)", value=20.5)
humidity = st.number_input("Humidity (%)", value=82.0)
ph = st.number_input("pH", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=202.0)




if st.button("🚀 Recommend Crops"):
    crop = recommend_crop([n, p, k, temp, humidity, ph, rainfall])
    st.success(f"🌱 Recommended Crop: **{crop}**")
# Done