# 💤 Project NIDRA  

An end-to-end **Machine Learning & Streamlit** project that predicts a **Sleep Score** based on health, lifestyle, and sleep-related inputs.  
It covers the full pipeline from **data collection → cleaning → EDA → model training → deployment**.

---

## 📌 Overview
- Collects and cleans a Kaggle sleep & lifestyle dataset  
- Performs **EDA** to explore patterns between lifestyle and sleep quality  
- Trains multiple ML models and builds an **Ensemble Voting Regressor**  
- Deploys an interactive **Streamlit web app** for real-time prediction  

---

## 🚀 Features
- ⚡ Handles missing values & scales features for better accuracy  
- 🧠 Uses multiple ML models (Linear Regression, SVR, Decision Tree, KNN, Gradient Boosting, Random Forest)  
- 🏆 Final prediction with **Ensemble Voting**  
- 🌐 Streamlit interface for user input & prediction  
- 📊 Categorizes Sleep Score into:  
  - **< 6 → Poor Sleep Quality (Bad)**  
  - **6 – 8 → Average to Good**  
  - **≥ 8 → Good Sleep Quality**

---

## 🗂️ Project Structure

```
Project_Nidra/
│
├── Data/
│   ├── Raw/                # Original Kaggle dataset
│   └── Cleaned/            # Cleaned dataset after preprocessing
│
├── Models/
│   ├── DecisionTree_model.pkl
│   ├── GradientBoosting_model.pkl
│   ├── KNN_model.pkl
│   ├── LinearRegression_model.pkl
│   ├── RandomForest_model.pkl
│   ├── SVR_model.pkl
│   └── scaler.pkl
│
├── Notebooks/
│   ├── Data_Cleaning.ipynb
│   ├── Data_Preprocessing.ipynb
│   ├── Model_Training.ipynb
│   └── Prediction.ipynb
│
├── Streamlit/
│   └── app.py              # Streamlit app for user interaction
│
├── requirements.txt
└── README.md

```


---

## 🔎 Workflow
1. **Data Collection** → Kaggle dataset (sleep & lifestyle data)  
2. **Data Cleaning & Processing** → Missing value handling, scaling, encoding  
3. **EDA** → Correlations, lifestyle vs sleep patterns visualization  
4. **Model Training** → Multiple regressors, final Ensemble Voting Regressor  
5. **Deployment** → Streamlit app for real-time Sleep Score prediction  

---

