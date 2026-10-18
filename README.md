<div align="center">

# 🚢 Titanic Survival Prediction System
### *An End-to-End Machine Learning Web Application & Real-Time Decision Simulator*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Deployment-Live-success?logo=vercel&logoColor=white)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Explore Features](#-key-features) • [About the Project](#-about-the-project) • [Model & Metrics](#-machine-learning-architecture) • [Run Locally](#-quick-start--local-setup) • [Deploy Online](#-1-click-cloud-deployment)

</div>

---

## 📖 About the Project

On April 15, 1912, the RMS Titanic sank in the North Atlantic Ocean after colliding with an iceberg during her maiden voyage. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it one of the deadliest peacetime commercial maritime disasters in modern history.

While there was an element of chance involved in surviving, historical records demonstrate that demographic, socioeconomic, and cabin location factors played a decisive role in who received access to lifeboats under the maritime protocol: *"Women and children first."*

### 🎯 Objective
This project bridges theoretical data science and production engineering. Rather than remaining an isolated Jupyter notebook, it delivers a **fully interactive, production-grade Machine Learning web application** that:
1. Employs a robust **Scikit-Learn Random Forest pipeline** to predict passenger survival probabilities.
2. Provides a **human-interpretable breakdown** of the socioeconomic and demographic factors influencing each prediction.
3. Incorporates **real-time presets** (such as *Rose DeWitt Bukater* and *Jack Dawson*) for instant historical simulation.
4. Delivers an **Exploratory Data Analysis (EDA)** suite analyzing historical trends across passenger classes, age demographics, and embarkation points.

---

## 🌟 Key Features

| Feature | Description |
| :--- | :--- |
| **🔮 Real-Time Survival Predictor** | Adjust passenger parameters (Ticket Class, Gender, Age, Family Members, Fare, and Embarkation Port) and view instant predictions with probability bars. |
| **⚡ Instant Preset Profiles** | One-click simulation buttons for famous profiles (e.g., *Rose DeWitt Bukater*, *Jack Dawson*, *1st Class Executives*, *Working-class Immigrants*). |
| **🧠 Interpretable AI / Factors** | Automatic risk assessment attributing predictions to maritime evacuation protocol, cabin deck proximity, and demographic priorities. |
| **📊 Visual Exploratory Analysis** | Dynamic bar charts and distributions illustrating survival rates across genders, classes, ports, and binned age brackets. |
| **🌲 Feature Importance Ranking** | Inspects the internal weights of the Random Forest model to understand what mattered most during inference. |
| **📱 Mobile & Desktop Optimized** | Fully responsive layout with adaptive typography, large touchscreen tap targets (44px min), and support for both Light and Dark modes. |

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────────────────────┐
│     Historical Kaggle Data      │
│        (data/train.csv)         │
└────────────────┬────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│             Pre-Processing Pipeline                    │
│  - Numerical: Median Imputer (Age, Fare, Pclass)       │
│  - Categorical: Mode Imputer + OneHotEncoder (Sex, Emb)│
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│             Model Training & Validation                │
│  - RandomForestClassifier (100 estimators, max_depth=7)│
│  - Stratified 80/20 Train-Test Validation              │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│            Serialized Production Artifact              │
│                     (model.joblib)                     │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│               Streamlit Web Interface                  │
│  - Live Predictor & Risk Assessment (app.py)           │
│  - Demographic Analysis & Feature Importance           │
│  - Cloud Hosted (Streamlit Community Cloud)            │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Machine Learning Architecture

The predictive engine uses an encapsulated Scikit-learn `Pipeline` paired with a `ColumnTransformer`:

### 1. Data Preprocessing
- **Numerical Features** (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`): Missing values are imputed using **Median Imputation** to prevent distortion from extreme fares and skewed age values.
- **Categorical Features** (`Sex`, `Embarked`): Missing values are resolved using **Mode Imputation**, followed by **One-Hot Encoding** with `drop='first'` to prevent multicollinearity.

### 2. Model Performance Benchmarks

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Validation Accuracy** | **~80.0% – 82.0%** | Evaluated on stratified test split |
| **Precision (Survived)** | **0.78** | Minimizes false positive survival claims |
| **Recall (Survived)** | **0.62 – 0.76** | Accurately identifies true survivors |
| **F1-Score** | **0.70 – 0.78** | Harmonic mean balancing precision and recall |

### 3. Feature Importance (Random Forest Weights)
1. **Gender (`Sex_male`)**: ~40.0% — The single most decisive factor ("Women and children first").
2. **Ticket Fare (`Fare`)**: ~19.8% — Proxy for cabin deck level and life vest access.
3. **Passenger Age (`Age`)**: ~14.9% — Prioritization of young children during lifeboat loading.
4. **Ticket Class (`Pclass`)**: ~11.8% — 1st Class cabins had direct boat-deck access.
5. **Family Relations (`SibSp`, `Parch`)**: ~9.5% — Small families coordinated successfully; large families faced severe delays.
6. **Embarkation Port (`Embarked`)**: ~3.8% — Slight correlation with socioeconomic ticket batches.

---

## 📂 Project Structure

```text
titanic-survival-prediction/
│
├── data/
│   └── train.csv            # Kaggle Titanic training dataset
├── app.py                   # Streamlit web application & UI logic
├── train_model.py           # Production model pipeline & serialization script
├── model.joblib             # Serialized trained model artifact
├── analysis.ipynb           # Exploratory data analysis notebook
├── requirements.txt         # Clean, platform-agnostic dependencies
├── .gitignore               # Ignored cache & environment directories
└── README.md                # Project documentation & reference
```

---

## 🚀 Quick Start & Local Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deepakpradhan1021/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Set up a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Retrain and rebuild model pipeline:**
   ```bash
   python train_model.py
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at **`http://localhost:8501`**.

---

## 🌐 1-Click Cloud Deployment

This project is optimized for deployment on **Streamlit Community Cloud**:

1. Fork or push this repository to your GitHub account: `https://github.com/deepakpradhan1021/titanic-survival-prediction`.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in using GitHub.
3. Click **"New app"** and fill in:
   - **Repository:** `deepakpradhan1021/titanic-survival-prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"** Your application will be live 24/7 with a shareable public URL.

---

## 🤝 Contributing

Contributions, issues, and feature suggestions are welcome!
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewFeature`).
3. Commit your Changes (`git commit -m "Add NewFeature"`).
4. Push to the Branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👤 Author

**Deepak Pradhan**
- 🐙 GitHub: [@deepakpradhan1021](https://github.com/deepakpradhan1021)
- 💼 Project: [titanic-survival-prediction](https://github.com/deepakpradhan1021/titanic-survival-prediction)