import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd

# === Load Models ===
disease_model = tf.keras.models.load_model('detetcion_modeltoday9-9.h5')
solution_model = joblib.load('solution_model.pkl')
solution_vectorizer = joblib.load('solution_vectorizer.pkl')
crop_model = joblib.load('crop_rf_model_rcc.pkl')
scaler = joblib.load('crop_scaler_rcc.pkl')
crop_label_encoder = joblib.load('crop_label_encoder_rcc.pkl')
# === Load Class Names ===
with open('classes.txt', 'r') as f:
    disease_classes = [line.strip() for line in f.readlines()]
print("✅ Disease Classes loaded:", disease_classes)

# === Load Disease-Solution CSV ===
disease_solution_df = pd.read_csv('plantvillage_disease_treatments.csv')
print("✅ CSV columns:", disease_solution_df.columns)

# === Disease Detection ===
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    preds = disease_model.predict(img_arr)
    predicted_class = disease_classes[np.argmax(preds)]
    return predicted_class

# === Solution Suggestion ===
def solu_crop(disease_name):
    disease_row = disease_solution_df[disease_solution_df['Disease'] == disease_name]
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


# === Crop Recommendation Function ===
def recommend_crop(input_features):
    input_scaled = scaler.transform([input_features])
    crop_index = crop_model.predict(input_scaled)[0]
    crop_name = crop_label_encoder.inverse_transform([crop_index])[0]
    return crop_name

# === Full Pipeline ===
def full_pipeline(img_path):
    print("\n🔍 Starting prediction pipeline...")

    predicted_disease = predict_disease(img_path)
    print(f"🦠 Predicted Disease: {predicted_disease}")

    crop, dis_name, symptoms, solution_text, treatment = solu_crop(predicted_disease)
    print(f"🌱 Crop: {crop}")
    print(f"📝 Disease Detail: {dis_name}")
    print(f"⚠️ Symptoms: {symptoms}")
    print(f"💊 Suggested Treatment: {solution_text}")
    print(f"🤖 Predicted Treatment Category: {treatment}")

# === Example Usage ===
if __name__ == "__main__":
   
   # 2. Recommend crop from soil data
    soil_data = [90, 42, 43, 20.5, 82.0, 6.5, 202.0]  # Example input
    recommended_crop = recommend_crop(soil_data)
    print(f"🌱 Recommended Crop: {recommended_crop}")