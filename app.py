import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .prediction-box-survived {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .prediction-box-perished {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    if not os.path.exists(model_path):
        # Fallback: train model on the fly if artifact not found
        from train_model import train_and_save_model
        train_and_save_model()
    return joblib.load(model_path)

@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "data", "train.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

try:
    artifact = load_model()
    model = artifact["model"]
    metrics = artifact.get("metrics", {})
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

df = load_dataset()

# Sidebar: Preset Profiles & Quick Navigation
st.sidebar.title("🚢 Passenger Profiles")
st.sidebar.markdown("Test famous passengers or customize your own input.")

preset = st.sidebar.selectbox(
    "Choose a Preset Profile:",
    [
        "Custom Input",
        "🌟 Rose DeWitt Bukater (1st Class)",
        "💔 Jack Dawson (3rd Class)",
        "💼 Business Executive (1st Class Male)",
        "👧 Young Girl with Family (2nd Class)",
        "⚓ Crew / Working Class (3rd Class Male)"
    ]
)

# Preset defaults
if preset == "🌟 Rose DeWitt Bukater (1st Class)":
    default_pclass = 1
    default_sex = "female"
    default_age = 17.0
    default_sibsp = 0
    default_parch = 1
    default_fare = 150.0
    default_embarked = "S"
elif preset == "💔 Jack Dawson (3rd Class)":
    default_pclass = 3
    default_sex = "male"
    default_age = 20.0
    default_sibsp = 0
    default_parch = 0
    default_fare = 8.05
    default_embarked = "S"
elif preset == "💼 Business Executive (1st Class Male)":
    default_pclass = 1
    default_sex = "male"
    default_age = 45.0
    default_sibsp = 0
    default_parch = 0
    default_fare = 85.0
    default_embarked = "C"
elif preset == "👧 Young Girl with Family (2nd Class)":
    default_pclass = 2
    default_sex = "female"
    default_age = 8.0
    default_sibsp = 1
    default_parch = 2
    default_fare = 26.25
    default_embarked = "S"
elif preset == "⚓ Crew / Working Class (3rd Class Male)":
    default_pclass = 3
    default_sex = "male"
    default_age = 28.0
    default_sibsp = 0
    default_parch = 0
    default_fare = 7.75
    default_embarked = "Q"
else:
    default_pclass = 3
    default_sex = "male"
    default_age = 28.0
    default_sibsp = 0
    default_parch = 0
    default_fare = 14.45
    default_embarked = "S"

st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")
if metrics:
    st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
    st.sidebar.metric("Precision", f"{metrics.get('precision', 0):.2f}")
    st.sidebar.metric("Recall", f"{metrics.get('recall', 0):.2f}")
st.sidebar.info("Model: Random Forest Classifier\nTrained with Scikit-learn Pipeline")

# Header Section
st.markdown('<div class="main-title">🚢 Titanic Survival Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An end-to-end Machine Learning web application predicting survival probabilities based on passenger demographics and ticket information.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Live Predictor", "📊 Exploratory Data Analysis", "ℹ️ Model & Project Details"])

with tab1:
    st.subheader("Enter Passenger Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pclass = st.selectbox(
            "Ticket Class (Pclass)",
            options=[1, 2, 3],
            index=[1, 2, 3].index(default_pclass),
            format_func=lambda x: {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}[x],
            help="Ticket class represents socioeconomic status."
        )
        sex = st.selectbox(
            "Gender",
            options=["male", "female"],
            index=["male", "female"].index(default_sex),
            format_func=lambda x: "Female" if x == "female" else "Male"
        )
    
    with col2:
        age = st.slider(
            "Age (Years)",
            min_value=0.42,
            max_value=80.0,
            value=float(default_age),
            step=1.0,
            help="Age in years. Infants under 1 year can be represented with decimals."
        )
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            index=["S", "C", "Q"].index(default_embarked),
            format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x]
        )
        
    with col3:
        sibsp = st.number_input(
            "Siblings / Spouses Aboard",
            min_value=0,
            max_value=8,
            value=int(default_sibsp),
            step=1,
            help="Number of siblings or spouses traveling with the passenger."
        )
        parch = st.number_input(
            "Parents / Children Aboard",
            min_value=0,
            max_value=6,
            value=int(default_parch),
            step=1,
            help="Number of parents or children traveling with the passenger."
        )
        fare = st.slider(
            "Ticket Fare (£ / GBP)",
            min_value=0.0,
            max_value=512.0,
            value=float(default_fare),
            step=1.0,
            help="Passenger fare paid for the journey."
        )

    # Predict Button
    predict_clicked = st.button("🔮 Predict Survival", use_container_width=True, type="primary")
    
    # Input DataFrame
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])
    
    # Perform prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    prob_survive = probabilities[1]
    prob_perish = probabilities[0]
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    res_col1, res_col2 = st.columns([1.2, 1])
    
    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box-survived">
                <h2 style="color: #065F46; margin-top:0;">🎉 Likely to Have Survived</h2>
                <p style="font-size: 1.1rem; color: #047857; margin-bottom: 0;">
                    The model predicts that this passenger had a <strong>high likelihood of surviving</strong> the disaster.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box-perished">
                <h2 style="color: #991B1B; margin-top:0;">⚠️ Unlikely to Have Survived</h2>
                <p style="font-size: 1.1rem; color: #B91C1C; margin-bottom: 0;">
                    The model predicts that this passenger had a <strong>low probability of surviving</strong> the disaster.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write(f"**Survival Probability:** `{prob_survive * 100:.1f}%`")
        st.progress(float(prob_survive))
        st.write(f"**Perish Probability:** `{prob_perish * 100:.1f}%`")

    with res_col2:
        st.markdown("#### Key Factor Summary")
        factors = []
        if sex == "female":
            factors.append("✅ **Gender (Female):** Female passengers had prioritized access to lifeboats ('Women and children first' maritime protocol).")
        else:
            factors.append("❌ **Gender (Male):** Male passengers suffered significantly lower survival rates (~19% overall).")
            
        if pclass == 1:
            factors.append("✅ **Class (1st Class):** 1st Class cabins were closer to the boat deck with highest survival rates (~63%).")
        elif pclass == 3:
            factors.append("❌ **Class (3rd Class):** 3rd Class cabins were located deep in lower decks with limited access to lifeboats.")
            
        if age < 12:
            factors.append("✅ **Age (Child):** Young children were given high priority during evacuation.")
            
        if fare > 50:
            factors.append("✅ **Fare:** High ticket fare strongly correlates with upper-deck staterooms.")

        for f in factors:
            st.markdown(f)

with tab2:
    st.subheader("Exploratory Data Analysis (EDA)")
    if df is not None:
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            st.markdown("##### Survival Rate by Gender")
            gender_survival = df.groupby("Sex")["Survived"].mean().reset_index()
            gender_survival["Survival Rate (%)"] = gender_survival["Survived"] * 100
            st.bar_chart(gender_survival.set_index("Sex")["Survival Rate (%)"])
            
            st.markdown("##### Survival Rate by Passenger Class")
            class_survival = df.groupby("Pclass")["Survived"].mean().reset_index()
            class_survival["Class"] = class_survival["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
            class_survival["Survival Rate (%)"] = class_survival["Survived"] * 100
            st.bar_chart(class_survival.set_index("Class")["Survival Rate (%)"])
            
        with eda_col2:
            st.markdown("##### Age Distribution of Passengers")
            st.line_chart(df["Age"].dropna().value_counts().sort_index())
            
            st.markdown("##### Survival Rate by Port of Embarkation")
            emb_survival = df.groupby("Embarked")["Survived"].mean().reset_index()
            emb_survival["Port"] = emb_survival["Embarked"].map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})
            emb_survival["Survival Rate (%)"] = emb_survival["Survived"] * 100
            st.bar_chart(emb_survival.set_index("Port")["Survival Rate (%)"])
            
        with st.expander("🔍 View Raw Training Dataset Sample"):
            st.dataframe(df.head(50))
    else:
        st.warning("Training dataset `data/train.csv` not available for display.")

with tab3:
    st.subheader("About the Project & Machine Learning Pipeline")
    st.markdown("""
    ### Project Architecture
    - **Dataset**: Kaggle Titanic: Machine Learning from Disaster.
    - **Algorithm**: `RandomForestClassifier` from Scikit-Learn.
    - **Feature Engineering & Imputation**:
      - Median imputation for numerical features (`Age`, `Fare`).
      - Mode imputation for categorical features (`Embarked`).
      - One-Hot Encoding for categorical features (`Sex`, `Embarked`).
    - **Pipeline**: Scikit-learn unified `ColumnTransformer` + `Pipeline`, exported to `model.joblib`.
    - **Web Framework**: Streamlit.
    
    ### How to Run Locally
    ```bash
    streamlit run app.py
    ```
    
    ### How to Deploy Online (Free 24/7 Hosting)
    1. Push this repository to GitHub.
    2. Go to [share.streamlit.io](https://share.streamlit.io).
    3. Select your repository `deepakpradhan1021/titanic-survival-prediction`.
    4. Set main file path to `app.py` and click **Deploy**!
    """)
    st.markdown("---")
    st.caption("Developed by Deepak Pradhan | Powered by Streamlit & Scikit-learn")
