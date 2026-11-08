# Titanic Survival Prediction & Web App

An end-to-end binary classification pipeline and interactive passenger survival simulator using Scikit-Learn and Streamlit based on the historical Titanic dataset.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data%20Analysis-Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Live Demo](https://titanic-survival-prediction-c3s4sq29gwey7kgtujcntq.streamlit.app/) • [About This Project](#-about-this-project) • [Pipeline & Results](#-machine-learning-pipeline--results) • [How to Run](#-how-to-run-locally) • [License](#-license)

---

## 🚀 Live Demo

Try the interactive passenger survival simulator in your browser:  
👉 **[Launch Titanic Survival Predictor](https://titanic-survival-prediction-c3s4sq29gwey7kgtujcntq.streamlit.app/)**

---

## 📖 About This Project

This project explores the classic Titanic dataset by building a complete machine learning workflow:
- **Data Analysis & Visualization:** Analyzing demographic factors that influenced survival rates during the 1912 disaster.
- **Pipeline Architecture:** Using `ColumnTransformer` and `Pipeline` to encapsulate preprocessing and estimator stages, preventing data leakage during cross-validation and testing.
- **Interactive Web App:** An intuitive Streamlit interface allowing users to adjust passenger parameters, test preset passenger profiles (e.g., *Rose*, *Jack*), and explore interactive EDA charts.
- **Model Serialization:** Saving the fitted pipeline with `joblib` for instant cloud serving.

---

## 🧠 Machine Learning Pipeline & Results

### 1. Preprocessing (`ColumnTransformer`)
- **Numerical Features** (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`): Imputed with median values.
- **Categorical Features** (`Sex`, `Embarked`): Imputed with the most frequent value (mode), followed by one-hot encoding (`drop='first'`).

### 2. Model Benchmark

Evaluated on an 80/20 stratified train-test split (`test_size=0.2`, `random_state=42`):

| Model | Test Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Classifier** | **0.82 (82%)** | **0.80** | **0.76** | **0.78** | **Selected for app** |
| **Logistic Regression** | 0.81 (81%) | 0.79 | 0.74 | 0.76 | Baseline |
| **Decision Tree Classifier** | 0.80 (80%) | 0.83 | 0.65 | 0.73 | Baseline |

### 3. Feature Importance Insights
The Random Forest model identified the following key predictors:
1. **Gender (`Sex`)**: ~40% importance (reflecting the "women and children first" maritime evacuation protocol).
2. **Ticket Fare (`Fare`)**: ~20% importance (correlated with deck location and cabin placement).
3. **Age (`Age`)**: ~15% importance.
4. **Ticket Class (`Pclass`)**: ~12% importance.
5. **Family Size (`SibSp`, `Parch`)**: ~9% importance.
6. **Port of Embarkation (`Embarked`)**: ~4% importance.

---

## 📂 Project Structure

```text
titanic-survival-prediction/
│
├── data/
│   └── train.csv            # Original training dataset
├── app.py                   # Streamlit web application
├── train_model.py           # Training & export script for model.joblib
├── model.joblib             # Serialized pipeline artifact
├── analysis.ipynb           # Initial exploratory data analysis notebook
├── requirements.txt         # Project dependencies
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 💻 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/deepakpradhan1021/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Retrain model (optional)

```bash
python train_model.py
```

### 5. Launch the Streamlit application

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
