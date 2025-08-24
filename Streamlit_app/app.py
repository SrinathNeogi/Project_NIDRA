import streamlit as st
import joblib
import numpy as np

# ==============================
# Custom CSS for Dark Theme
# ==============================
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .title {
        font-size: 38px;
        font-weight: bold;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #8b949e;
        margin-bottom: 25px;
    }
    div.stButton > button:first-child {
        background-color: #238636;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 12px 24px;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #2ea043;
        transform: scale(1.05);
    }
    .sleep-score {
        background-color: #161b22;
        border-left: 6px solid #f39c12;
        padding: 12px;
        font-size: 20px;
        color: #f1c40f;
        font-weight: bold;
        border-radius: 8px;
    }
    .prediction-interpret {
        background-color: #161b22;
        border-left: 6px solid #9b59b6;
        padding: 12px;
        font-size: 16px;
        color: #c8a2c8;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Title
# ==============================
st.markdown('<div class="title">🌙 Project <span style="color:#e63946;">NIDRA</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sleep Score Prediction App</div>', unsafe_allow_html=True)

# ==============================
# Load Scaler and Models
# ==============================
scalers = joblib.load("scaler.pkl")

model_names = {
    "GradientBoosting": "GradientBoosting_model.pkl",
    "KNN": "KNN_model.pkl",
    "RandomForest": "RandomForest_model.pkl",
    "DecisionTree": "DecisionTree_model.pkl",
    "LinearRegression": "LinearRegression_model.pkl",
    "SVR": "SVR_model.pkl"
}

models = {name: joblib.load(path) for name, path in model_names.items()}

# ==============================
# Encoders (mapping)
# ==============================
gender_map = {"Male": 1, "Female": 0}
bmi_map = {"Normal": 0, "Obese": 1, "Overweight": 2, "Underweight": 3}
sleep_map = {"Normal": 1, "Insomnia": 0, "Sleep Apnea": 2}

numerical_cols = [
    "Age", "Sleep Duration", "Physical Activity Level",
    "Stress Level", "Heart Rate", "Daily Steps",
    "Systolic_BP", "Diastolic_BP"
]

# ==============================
# Preprocess input
# ==============================
def preprocess_input(data):
    gender = gender_map[data['Gender']]
    bmi = bmi_map[data['BMI Category']]
    sleep = sleep_map[data['Sleep Disorder']]

    numerical_scaled = []
    for col in numerical_cols:
        sc = scalers[col]
        scaled_val = sc.transform([[data[col]]])[0][0]
        numerical_scaled.append(scaled_val)

    features = np.hstack([gender, bmi, sleep, numerical_scaled])
    return features.reshape(1, -1)

# ==============================
# Prediction Logic
# ==============================
def predict_sleep_score(features):
    predictions = {name: model.predict(features)[0] for name, model in models.items()}
    
    linear_pred = predictions["LinearRegression"]
    avg_pred = np.mean(list(predictions.values()))

    # Rule: if Linear Regression < 5, show it mandatorily
    if linear_pred < 5:
        return linear_pred
    else:
        return avg_pred

# ==============================
# Streamlit UI
# ==============================
st.write("### Enter your health and lifestyle details to predict your sleep score:")

gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.selectbox("BMI Category", ["Normal", "Obese", "Overweight", "Underweight"])
sleep_disorder = st.selectbox("Sleep Disorder", ["Normal", "Insomnia", "Sleep Apnea"])

age = st.number_input("Age", min_value=10, max_value=100, value=25)
sleep_duration1 = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, 0.5)
sleep_duration = 12 - sleep_duration1
activity = st.slider("Physical Activity Level (minutes/day)", 0, 300, 30)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)
systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP", min_value=50, max_value=120, value=80)

# Prediction button
if st.button("✨ Predict My Sleep Score ✨"):
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
    final_pred = predict_sleep_score(features)

    st.markdown("---")

    # Final Prediction
    st.markdown(f"""
        <div class="sleep-score">🌟 Sleep Score: {final_pred:.2f}</div>
    """, unsafe_allow_html=True)

    if final_pred >= 8:
        category = "Good"
        advice = "Your sleep score is good! Keep maintaining your healthy habits."
    elif 6 <= final_pred < 8:
        category = "Average to Good"
        advice = "Your sleep score is average to good. With small improvements, it can become excellent."
    else:
        category = "Bad"
        advice = "Your sleep score is low. Consider improving your sleep habits and lifestyle."

    st.markdown(f"""
        <div class="prediction-interpret">
        📝 Category: <b>{category}</b><br>
        {advice}
        </div>
    """, unsafe_allow_html=True)
