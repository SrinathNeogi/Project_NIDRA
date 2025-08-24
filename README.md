# Project NIDRA 💤

**NIDRA** is an end-to-end Machine Learning project that predicts a **Sleep Score** based on health, lifestyle, and sleep-related inputs.  
It covers the full pipeline from **data collection** → **data cleaning** → **EDA** → **model training & evaluation** → **deployment as a Streamlit web app**.

---

## 📂 Project Structure

Project_Nidra/
│
├── Data/
│ ├── Raw/ # Original Kaggle dataset
│ └── Cleaned/ # Cleaned dataset after preprocessing
│
├── Models/
│ ├── models.pkl # Trained ensemble model
│ └── scaler.pkl # Saved StandardScaler objects
│
├── Notebooks/
│ ├── Data_Cleaning.ipynb
│ ├── Data_Preprocessing.ipynb
│ ├── Model_Training.ipynb
│ └── Prediction.ipynb
│
├── Streamlit/
│ └── app.py # Streamlit app for user interaction
│
├── requirements.txt
└── README.md


---

## 📊 Workflow

1. **Data Collection**  
   - Dataset collected from **Kaggle** (sleep & lifestyle dataset).  

2. **Data Cleaning & Processing**  
   - Handling missing values  
   - Scaling numerical features using `StandardScaler`  
   - Encoding categorical features  

3. **Exploratory Data Analysis (EDA)**  
   - Visualizing relationships between lifestyle factors and sleep quality  
   - Identifying correlations & insights  

4. **Model Training**  
   - Tried multiple regression models:
     - Linear Regression  
     - Support Vector Regressor (SVR)  
     - Decision Tree  
     - K-Nearest Neighbors (KNN)  
     - Gradient Boosting Regressor  
     - Random Forest Regressor  
   - Final model is an **Ensemble Voting Regressor**  

5. **Deployment**  
   - Built an interactive **Streamlit web app**  
   - Users can enter lifestyle/health data → get a **Predicted Sleep Score**  
   - Categorized into:
     - **< 6** → Poor Sleep Quality (Bad)  
     - **6 – 8** → Average to Good  
     - **≥ 8** → Good Sleep Quality  

---

