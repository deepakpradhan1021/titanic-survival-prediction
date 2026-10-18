import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration with responsive layout
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="auto"  # "auto" ensures sidebar does not obscure the screen on mobile devices
)

# Responsive & Adaptive Styling (Supports Light & Dark Modes and Mobile Viewports)
st.markdown("""
<style>
    /* Responsive Title & Subtitle */
    .hero-title {
        font-size: clamp(1.6rem, 4vw, 2.3rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: clamp(0.95rem, 2vw, 1.1rem);
        opacity: 0.85;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }
    
    /* Result Cards with Adaptive Backgrounds */
    .prediction-card-survived {
        padding: 1.2rem;
        border-radius: 12px;
        background-color: rgba(16, 185, 129, 0.12);
        border: 2px solid #10B981;
        margin: 1rem 0;
    }
    .prediction-card-perished {
        padding: 1.2rem;
        border-radius: 12px;
        background-color: rgba(239, 68, 68, 0.12);
        border: 2px solid #EF4444;
        margin: 1rem 0;
    }
    .result-heading {
        margin: 0 0 0.5rem 0;
        font-size: clamp(1.3rem, 3vw, 1.7rem);
        font-weight: 700;
    }
    
    /* Mobile-optimized touch buttons */
    div[data-testid="stButton"] button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 44px; /* Accessible mobile touch target size */
    }
    
    /* Info chips / badges */
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(59, 130, 246, 0.15);
        color: #2563EB;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Preset Passenger Configurations
PRESETS = {
    "🌟 Rose DeWitt Bukater": {
        "pclass": 1,
        "sex": "female",
        "age": 17.0,
        "sibsp": 0,
        "parch": 1,
        "fare": 150.0,
        "embarked": "S",
        "desc": "1st Class aristocratic passenger (Movie Character)"
    },
    "💔 Jack Dawson": {
        "pclass": 3,
        "sex": "male",
        "age": 20.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 8.05,
        "embarked": "S",
        "desc": "3rd Class artist traveling solo (Movie Character)"
    },
    "💼 1st Class Executive": {
        "pclass": 1,
        "sex": "male",
        "age": 45.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 85.0,
        "embarked": "C",
        "desc": "Wealthy businessman embarking from Cherbourg"
    },
    "👧 2nd Class Child": {
        "pclass": 2,
        "sex": "female",
        "age": 8.0,
        "sibsp": 1,
        "parch": 2,
        "fare": 26.25,
        "embarked": "S",
        "desc": "Young girl traveling with parents from Southampton"
    },
    "⚓ 3rd Class Immigrant": {
        "pclass": 3,
        "sex": "male",
        "age": 28.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 7.75,
        "embarked": "Q",
        "desc": "Working-class solo traveler from Queenstown"
    }
}

# Initialize Session State
default_values = {
    "pclass": 3,
    "sex": "male",
    "age": 28.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 14.45,
    "embarked": "S",
    "active_preset": "Custom"
}
for k, v in default_values.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_preset(name):
    if name in PRESETS:
        p = PRESETS[name]
        st.session_state["pclass"] = p["pclass"]
        st.session_state["sex"] = p["sex"]
        st.session_state["age"] = p["age"]
        st.session_state["sibsp"] = p["sibsp"]
        st.session_state["parch"] = p["parch"]
        st.session_state["fare"] = p["fare"]
        st.session_state["embarked"] = p["embarked"]
        st.session_state["active_preset"] = name

# Cached Model Loader
@st.cache_resource
def load_model_artifact():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    if not os.path.exists(model_path):
        from train_model import train_and_save_model
        train_and_save_model(model_path=model_path)
    return joblib.load(model_path)

# Cached Dataset Loader
@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "data", "train.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

try:
    artifact = load_model_artifact()
    model = artifact["model"]
    metrics = artifact.get("metrics", {})
except Exception as e:
    st.error(f"Error initializing model: {e}")
    st.stop()

df = load_dataset()

# Sidebar: Quick Info & Metrics
with st.sidebar:
    st.title("🚢 Titanic ML")
    st.markdown("Machine Learning Model for Titanic Survival Prediction.")
    
    st.subheader("Model Performance")
    if metrics:
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        col_m2.metric("F1-Score", f"{metrics.get('f1', 0):.2f}")
        col_m1.metric("Precision", f"{metrics.get('precision', 0):.2f}")
        col_m2.metric("Recall", f"{metrics.get('recall', 0):.2f}")
    
    st.markdown("---")
    st.subheader("Presets (Sidebar)")
    sidebar_preset = st.selectbox(
        "Load Preset Profile:",
        ["Select a preset..."] + list(PRESETS.keys())
    )
    if sidebar_preset != "Select a preset..." and sidebar_preset != st.session_state["active_preset"]:
        apply_preset(sidebar_preset)
        st.rerun()
        
    st.info("💡 **Tip**: You can customize any input after choosing a preset.")

# Hero Header
st.markdown('<div class="hero-title">🚢 Titanic Passenger Survival Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Predict the survival probability of passengers on the RMS Titanic using a production Random Forest Classifier pipeline trained on Kaggle demographic and ticket data.</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Live Predictor", "📊 Exploratory Data Analysis (EDA)", "ℹ️ Model Details & Methodology"])

# ==========================================
# TAB 1: LIVE PREDICTOR
# ==========================================
with tab1:
    # Quick Preset Bar (Mobile friendly!)
    st.markdown("##### ⚡ Quick Presets (One-Click Test)")
    preset_cols = st.columns(len(PRESETS))
    for i, (name, details) in enumerate(PRESETS.items()):
        if preset_cols[i].button(name, key=f"btn_preset_{i}", use_container_width=True):
            apply_preset(name)
            st.rerun()

    if st.session_state["active_preset"] in PRESETS:
        st.caption(f"ℹ️ Active Preset: **{st.session_state['active_preset']}** — {PRESETS[st.session_state['active_preset']]['desc']}")

    st.markdown("---")
    st.subheader("Passenger Parameters")

    # Inputs layout: responsive 3 columns
    inp_col1, inp_col2, inp_col3 = st.columns(3)

    with inp_col1:
        pclass = st.selectbox(
            "Ticket Class (Pclass)",
            options=[1, 2, 3],
            index=[1, 2, 3].index(st.session_state["pclass"]),
            format_func=lambda x: {1: "1st Class (Upper Deck)", 2: "2nd Class (Middle Deck)", 3: "3rd Class (Lower Deck)"}[x],
            help="Ticket class represents socioeconomic status and proximity to the lifeboat deck."
        )
        sex = st.selectbox(
            "Passenger Gender",
            options=["female", "male"],
            index=["female", "male"].index(st.session_state["sex"]),
            format_func=lambda x: "Female" if x == "female" else "Male"
        )

    with inp_col2:
        age = st.slider(
            "Age (Years)",
            min_value=0.5,
            max_value=80.0,
            value=float(st.session_state["age"]),
            step=0.5,
            help="Age in years. Historical protocol prioritized children."
        )
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            index=["S", "C", "Q"].index(st.session_state["embarked"]),
            format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x],
            help="Port where the passenger boarded the ship."
        )

    with inp_col3:
        sibsp = st.number_input(
            "Siblings / Spouses Aboard (SibSp)",
            min_value=0,
            max_value=8,
            value=int(st.session_state["sibsp"]),
            step=1,
            help="Number of siblings or spouses traveling with the passenger."
        )
        parch = st.number_input(
            "Parents / Children Aboard (Parch)",
            min_value=0,
            max_value=6,
            value=int(st.session_state["parch"]),
            step=1,
            help="Number of parents or children traveling with the passenger."
        )
        fare = st.slider(
            "Ticket Fare (£ / GBP)",
            min_value=0.0,
            max_value=512.0,
            value=float(st.session_state["fare"]),
            step=1.0,
            help="Passenger fare paid in British Pounds (historical)."
        )

    # Sync state when adjusted
    st.session_state["pclass"] = pclass
    st.session_state["sex"] = sex
    st.session_state["age"] = age
    st.session_state["sibsp"] = sibsp
    st.session_state["parch"] = parch
    st.session_state["fare"] = fare
    st.session_state["embarked"] = embarked

    # Prepare DataFrame for Prediction
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    # Run Prediction
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    prob_survive = float(probs[1])
    prob_perish = float(probs[0])

    st.markdown("---")
    st.subheader("Prediction & Risk Assessment")

    out_col1, out_col2 = st.columns([1.1, 1])

    with out_col1:
        if pred == 1:
            st.markdown(f"""
            <div class="prediction-card-survived">
                <div class="result-heading" style="color: #059669;">🎉 LIKELY TO SURVIVE</div>
                <div style="font-size: 1.05rem; line-height: 1.5; color: #065F46;">
                    The model estimates a <strong>high probability of survival</strong> under historical evacuation conditions.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card-perished">
                <div class="result-heading" style="color: #DC2626;">⚠️ UNLIKELY TO SURVIVE</div>
                <div style="font-size: 1.05rem; line-height: 1.5; color: #991B1B;">
                    The model estimates a <strong>low probability of survival</strong> due to demographics, class, or cabin location.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"**Survival Probability:** `{prob_survive * 100:.1f}%`")
        st.progress(prob_survive)
        st.markdown(f"**Perish Probability:** `{prob_perish * 100:.1f}%`")

    with out_col2:
        st.markdown("#### Influencing Factors")
        factors = []
        
        # Gender Analysis
        if sex == "female":
            factors.append("✅ **Gender (Female):** Highest survival priority. ~74% of female passengers survived due to the 'Women and children first' order.")
        else:
            factors.append("❌ **Gender (Male):** Lowest survival priority. Only ~19% of male passengers survived.")

        # Class Analysis
        if pclass == 1:
            factors.append("✅ **1st Class Ticket:** Staterooms were on upper decks with rapid and direct access to lifeboats (~63% survival rate).")
        elif pclass == 2:
            factors.append("⚖️ **2nd Class Ticket:** Moderate survival rate (~47%), favorable primarily for women and children.")
        else:
            factors.append("❌ **3rd Class Ticket:** Cabins were on lower decks with complex escape routes and locked gates (~24% survival rate).")

        # Age Analysis
        if age < 12:
            factors.append("✅ **Young Child:** Children were given priority boarding onto lifeboats.")
        elif age > 60:
            factors.append("⚠️ **Senior Passenger:** Older passengers faced physical mobility challenges during the freezing night evacuation.")

        # Family Size Analysis
        total_family = sibsp + parch
        if total_family == 0:
            factors.append("⚖️ **Solo Passenger:** No family members to find or assist, but lacked coordination support.")
        elif 1 <= total_family <= 3:
            factors.append("✅ **Small Family (1-3):** Optimal survival balance; families were able to board lifeboats together.")
        else:
            factors.append("❌ **Large Family (>3):** Large families had difficulty staying together during the chaos, resulting in lower survival.")

        # Fare Analysis
        if fare > 50:
            factors.append("✅ **High Fare:** Strongly correlated with premium 1st Class cabin locations.")
        elif fare < 10:
            factors.append("❌ **Low Fare:** Correlated with lower-tier 3rd Class steerage berths.")

        for factor in factors:
            st.markdown(factor)

# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab2:
    st.subheader("Historical Titanic Dataset Analysis")
    st.markdown("Gain insights from the historical Kaggle Titanic dataset (`train.csv`) to understand survival distributions.")

    if df is not None:
        eda1, eda2 = st.columns(2)

        with eda1:
            st.markdown("##### 👥 Survival Rate by Gender")
            gender_stats = df.groupby("Sex")["Survived"].mean().reset_index()
            gender_stats["Survival Rate (%)"] = gender_stats["Survived"] * 100
            st.bar_chart(gender_stats.set_index("Sex")["Survival Rate (%)"], color="#3B82F6")

            st.markdown("##### 🎫 Survival Rate by Passenger Class")
            class_stats = df.groupby("Pclass")["Survived"].mean().reset_index()
            class_stats["Class Name"] = class_stats["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
            class_stats["Survival Rate (%)"] = class_stats["Survived"] * 100
            st.bar_chart(class_stats.set_index("Class Name")["Survival Rate (%)"], color="#10B981")

        with eda2:
            st.markdown("##### ⚓ Survival Rate by Port of Embarkation")
            port_stats = df.groupby("Embarked")["Survived"].mean().reset_index()
            port_stats["Port Name"] = port_stats["Embarked"].map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})
            port_stats["Survival Rate (%)"] = port_stats["Survived"] * 100
            st.bar_chart(port_stats.set_index("Port Name")["Survival Rate (%)"], color="#8B5CF6")

            st.markdown("##### 🎂 Survival Rate by Age Group")
            # Create readable age bins
            age_bins = [0, 12, 19, 35, 55, 100]
            age_labels = ["0-12 (Child)", "13-19 (Teen)", "20-35 (Young Adult)", "36-55 (Adult)", "56+ (Senior)"]
            df_age = df.dropna(subset=["Age"]).copy()
            df_age["AgeGroup"] = pd.cut(df_age["Age"], bins=age_bins, labels=age_labels, right=True)
            age_group_stats = df_age.groupby("AgeGroup", observed=False)["Survived"].mean().reset_index()
            age_group_stats["Survival Rate (%)"] = age_group_stats["Survived"] * 100
            st.bar_chart(age_group_stats.set_index("AgeGroup")["Survival Rate (%)"], color="#F59E0B")

        st.markdown("---")
        st.markdown("##### 🌲 Model Feature Importance (Random Forest)")
        # Extract feature importances
        try:
            preproc = model.named_steps["preprocessor"]
            clf = model.named_steps["classifier"]
            raw_features = preproc.get_feature_names_out()
            
            clean_name_map = {
                "cat__Sex_male": "Gender (Male)",
                "num__Fare": "Ticket Fare",
                "num__Age": "Passenger Age",
                "num__Pclass": "Passenger Class (Pclass)",
                "num__SibSp": "Siblings / Spouses Aboard",
                "num__Parch": "Parents / Children Aboard",
                "cat__Embarked_S": "Port Southampton",
                "cat__Embarked_Q": "Port Queenstown"
            }
            clean_names = [clean_name_map.get(f, f) for f in raw_features]
            fi_df = pd.DataFrame({
                "Feature": clean_names,
                "Relative Importance": clf.feature_importances_
            }).sort_values("Relative Importance", ascending=True)

            st.bar_chart(fi_df.set_index("Feature")["Relative Importance"], horizontal=True, color="#EC4899")
        except Exception as fi_err:
            st.info(f"Feature importance display: {fi_err}")

        with st.expander("🔍 Explore Raw Dataset (First 100 Rows)"):
            st.dataframe(df.head(100))
    else:
        st.warning("Training data file `data/train.csv` not detected.")

# ==========================================
# TAB 3: MODEL & METHODOLOGY DETAILS
# ==========================================
with tab3:
    st.subheader("Model Specifications & Architecture")
    st.markdown("""
    ### 🔬 Pipeline Details
    This system applies a standard machine learning engineering architecture:
    
    1. **Preprocessing Pipeline (`ColumnTransformer`)**:
       - **Numerical Features** (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`): Missing values are imputed using **Median Imputation**, robust to extreme outliers in fares and ages.
       - **Categorical Features** (`Sex`, `Embarked`): Missing values are imputed using **Most Frequent (Mode) Imputation**, followed by **One-Hot Encoding** (`drop='first'`).
    2. **Classifier (`RandomForestClassifier`)**:
       - An ensemble of 100 decision trees (`n_estimators=100`, `max_depth=7`, `random_state=42`) to mitigate overfitting while capturing non-linear interactions.
    3. **Serialization**:
       - The complete pipeline is saved in `model.joblib` using `joblib`, making inferences self-contained without needing raw training data in production.

    ### 📈 Key Evaluation Metrics
    """)
    if metrics:
        col_eval1, col_eval2, col_eval3, col_eval4 = st.columns(4)
        col_eval1.metric("Validation Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        col_eval2.metric("Precision (Survived)", f"{metrics.get('precision', 0):.2f}")
        col_eval3.metric("Recall (Survived)", f"{metrics.get('recall', 0):.2f}")
        col_eval4.metric("F1-Score", f"{metrics.get('f1', 0):.2f}")

    st.markdown("""
    ---
    ### 📱 Cross-Device Optimization
    - **Mobile Viewports**: Optimized layouts with flexible column stacking, large touch targets (44px min-height), and collapsible sidebars.
    - **Desktop Viewports**: Expanded wide view with multi-column parameters and responsive charts.
    - **Theme Compatibility**: Tested and verified across both Light Mode and Dark Mode environments.
    
    ### 🌐 Deployment Architecture
    - **Host**: Streamlit Community Cloud (connected directly to GitHub repository).
    - **CI/CD**: Automatically redeploys on every `git push origin main`.
    """)

st.markdown("---")
st.caption("Developed by Deepak Pradhan | Built with Python, Scikit-learn, and Streamlit")
