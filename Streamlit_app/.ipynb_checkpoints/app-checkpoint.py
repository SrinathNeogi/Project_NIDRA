import streamlit as st
import joblib
import numpy as np

# ==============================
# Load Scaler and Models
# ==============================
scalers = joblib.load("scaler.pkl")

model_names = [
    "GradientBoosting_model.pkl", "KNN_model.pkl",
    "RandomForest_model.pkl", "DecisionTree_model.pkl",
    "LinearRegression_model.pkl", "SVR_model.pkl"
]

models = [joblib.load(name) for name in model_names]

# ==============================
# Encoders (mapping)
# ==============================
gender_map = {"Male": 1, "Female": 0}
bmi_map = {"Normal": 0, "Obese": 1, "Overweight": 2, "Underweight": 3}
sleep_map = {"Normal": 1, "Insomnia": 0, "Sleep Apnea": 2}

# Numerical columns order
numerical_cols = [
    "Age", "Sleep Duration", "Physical Activity Level",
    "Stress Level", "Heart Rate", "Daily Steps",
    "Systolic_BP", "Diastolic_BP"
]

# ==============================
# Preprocess input
# ==============================
def preprocess_input(data):
    # Encode categorical
    gender = gender_map[data['Gender']]
    bmi = bmi_map[data['BMI Category']]
    sleep = sleep_map[data['Sleep Disorder']]

    # Scale numerical with respective scalers
    numerical_scaled = []
    for col in numerical_cols:
        sc = scalers[col]
        scaled_val = sc.transform([[data[col]]])[0][0]
        numerical_scaled.append(scaled_val)

    # Final feature vector
    features = np.hstack([gender, bmi, sleep, numerical_scaled])
    return features.reshape(1, -1)

# ==============================
# Prediction
# ==============================
def predict_sleep_score(features):
    predictions = [model.predict(features)[0] for model in models]
    final_pred = np.mean(predictions)
    return predictions, final_pred

# ==============================
# Streamlit UI
# ==============================
st.title("Project NIDRA - Sleep Score Prediction App")
st.write("Enter your health and lifestyle details to predict your sleep score.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.selectbox("BMI Category", ["Normal", "Obese", "Overweight", "Underweight"])
sleep_disorder = st.selectbox("Sleep Disorder", ["Normal", "Insomnia", "Sleep Apnea"])

age = st.number_input("Age", min_value=10, max_value=100, value=25)
sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, 0.5)
activity = st.slider("Physical Activity Level (minutes/day)", 0, 300, 30)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)
systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP", min_value=50, max_value=120, value=80)

# Prediction button
if st.button("Predict Sleep Score"):
    user_data = {
        "Gender": gender,
        "BMI Category": bmi,
        "Sleep Disorder": sleep_disorder,
        "Age": age,
        "Sleep Duration": sleep_duration,
        "Physical Activity Level": activity,
        "Stress Level": stress,
        "Heart Rate": heart_rate,
        "Daily Steps": steps,
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic
    }

    features = preprocess_input(user_data)
    all_preds, avg_pred = predict_sleep_score(features)

    # Display results
    st.subheader("🔍 Model Predictions")
    st.write({name: round(pred, 2) for name, pred in zip(model_names, all_preds)})

    st.subheader("📊 Final Averaged Prediction")
    st.write(f"**Sleep Score: {avg_pred:.2f}**")

    # Interpretation
    if avg_pred >= 8:
        category = "Good"
        advice = "Your sleep score is good! Keep maintaining your healthy habits."
    elif 6 <= avg_pred < 8:
        category = "Average to Good"
        advice = "Your sleep score is average to good. With small improvements, it can become excellent."
    else:
        category = "Bad"
        advice = "Your sleep score is low. Consider improving your sleep habits and lifestyle."

    st.subheader("📝 Interpretation")
    st.write(f"Category: **{category}**")
    st.write(advice)
