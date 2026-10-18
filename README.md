# 🚢 Titanic Survival Prediction & Interactive Web App

A production-ready machine learning web application that predicts passenger survival on the Titanic with real-time probabilistic insights and interactive Exploratory Data Analysis (EDA).

---

## 🌟 Live Features

- **🔮 Interactive Real-Time Predictor**: Select passenger demographics, ticket class, fare, family size, and embarkation port to receive instant survival predictions and probability scores.
- **🎭 Preset Profiles**: Instant one-click simulation for famous passengers (e.g., *Rose DeWitt Bukater*, *Jack Dawson*, and typical demographic archetypes).
- **📊 Exploratory Data Analysis**: Visual breakdown of historical survival rates across gender, passenger class, age distribution, and embarkation ports.
- **🧠 Production ML Pipeline**: Built with Scikit-learn unified `ColumnTransformer` + `Pipeline`, combining median/mode imputation, one-hot encoding, and a tuned `RandomForestClassifier`.

---

## 🛠️ Technologies Used

- **Web Application**: Streamlit
- **Machine Learning**: Scikit-Learn (`RandomForestClassifier`, `Pipeline`, `ColumnTransformer`)
- **Data Manipulation & Serialization**: Pandas, NumPy, Joblib
- **Visualization**: Altair, Matplotlib, Seaborn
- **Development**: Python 3.13+, Jupyter Notebook

---

## 📂 Project Structure

```text
titanic-survival-prediction/
│
├── data/
│   └── train.csv            # Training dataset (Kaggle Titanic)
├── app.py                   # Interactive Streamlit Web Application
├── train_model.py           # Model training and artifact pipeline script
├── model.joblib             # Serialized trained model pipeline
├── analysis.ipynb           # Exploratory data analysis notebook
├── requirements.txt         # Production dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

## 🚀 Running the Project Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deepakpradhan1021/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
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

4. **(Optional) Train or regenerate the model artifact:**
   ```bash
   python train_model.py
   ```

5. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🌐 Deploying Live to the Web (Streamlit Community Cloud)

You can deploy this project live for free in under 2 minutes:

1. Push your updated code to GitHub:
   ```bash
   git add .
   git commit -m "Add interactive web app and model artifact for live deployment"
   git push origin main
   ```
2. Navigate to [Streamlit Community Cloud](https://share.streamlit.io).
3. Sign in with GitHub.
4. Click **New app** and configure:
   - **Repository**: `deepakpradhan1021/titanic-survival-prediction`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!** Your application will be live at `https://<your-custom-name>.streamlit.app`.

---

## 📊 Model & Performance

- **Model Type**: Random Forest Classifier (`n_estimators=100, max_depth=7`)
- **Evaluation Accuracy**: ~80% - 82%
- **Target Variable**:
  - `Survived = 1` → Passenger survived
  - `Survived = 0` → Passenger did not survive
- **Key Determinants**: Gender (`Sex`), Passenger Class (`Pclass`), and Ticket Fare (`Fare`).

---

## 👤 Author

**Deepak Pradhan**  
- GitHub: [@deepakpradhan1021](https://github.com/deepakpradhan1021)