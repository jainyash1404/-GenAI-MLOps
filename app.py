import streamlit as st
import pandas as pd
import joblib
import os
import json
import textwrap
import html

from datetime import datetime
from genai_recommendation import generate_recommendations


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# HTML RENDERING HELPER
# =====================================================

def render_html(content):
    """
    Render HTML directly instead of displaying HTML tags
    as code or markdown.
    """
    content = textwrap.dedent(content).strip()

    st.html(content)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080B16;
    color: #F8FAFC;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #10182B,
        #0B1020
    );
    border-right: 1px solid #263653;
}

[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

[data-testid="stSidebar"] p {
    color: #94A3B8;
}

/* HEADINGS */

h1, h2, h3, h4 {
    color: #F8FAFC !important;
    font-weight: 800 !important;
}

p, label {
    color: #AAB8CE;
}

/* HERO */

.hero {
    background: linear-gradient(
        120deg,
        #172C52,
        #302267,
        #171C3B
    );

    border: 1px solid #46558A;
    border-radius: 22px;
    padding: 42px;
    margin-bottom: 35px;
    position: relative;
    overflow: hidden;
}

.hero-badge {
    display: inline-block;
    padding: 8px 16px;
    background: rgba(74,222,128,0.12);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 30px;
    color: #4ADE80;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 20px;
}

.hero-title {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero-description {
    color: #C1CDE5;
    font-size: 15px;
    line-height: 1.8;
}

/* KPI CARDS */

.kpi-card {
    background: linear-gradient(
        145deg,
        #192640,
        #111A2E
    );

    border: 1px solid #304365;
    border-radius: 18px;
    padding: 24px;
    min-height: 150px;
    transition: all 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    border-color: #818CF8;
}

.kpi-title {
    color: #9EADC7;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.kpi-value {
    color: #60A5FA;
    font-size: 31px;
    font-weight: 800;
    margin-top: 14px;
}

.kpi-sub {
    color: #71819E;
    font-size: 12px;
    margin-top: 7px;
}

/* RESULT CARDS */

.result-card {
    background: linear-gradient(
        135deg,
        #1A3158,
        #211D50
    );

    border: 1px solid #465C96;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    min-height: 190px;
}

.result-label {
    color: #B6C4E4;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}

.result-score {
    font-size: 55px;
    font-weight: 800;
    color: #60A5FA;
    margin: 15px 0;
}

/* AI CARD */

.ai-card {
    background: linear-gradient(
        120deg,
        #201D4C,
        #171D38
    );

    border: 1px solid #51458D;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 20px;
}

/* SECTION LABEL */

.section-label {
    color: #818CF8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 15px;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(
        100deg,
        #2563EB,
        #7C3AED
    );

    color: white !important;
    border: none;
    border-radius: 11px;
    height: 47px;
    font-weight: 700;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 22px rgba(99,102,241,0.4);
}

/* INPUTS */

.stSlider label {
    color: #CBD5E1 !important;
    font-weight: 600;
}

[data-testid="stSlider"] [role="slider"] {
    background: #818CF8;
}

/* METRICS */

[data-testid="stMetric"] {
    background: #151F34;
    border: 1px solid #2D405F;
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stMetricValue"] {
    color: #60A5FA !important;
    font-weight: 800;
}

[data-testid="stMetricLabel"] {
    color: #AAB8CE !important;
}

/* STATUS */

.status-card {
    background: #102E29;
    border: 1px solid #236B52;
    border-radius: 14px;
    padding: 20px;
    color: #4ADE80;
}

/* TABLE */

[data-testid="stDataFrame"] {
    border: 1px solid #293958;
    border-radius: 12px;
}

/* TABS */

.stTabs [data-baseweb="tab"] {
    color: #A5B4C9;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #818CF8 !important;
}

/* DIVIDER */

hr {
    border-color: #293958;
}

/* FOOTER */

.footer {
    text-align: center;
    padding: 30px;
    color: #71819E;
    font-size: 12px;
    line-height: 2;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# FILE CONFIGURATION
# =====================================================

LOG_FILE = "logs/predictions.csv"
STATS_FILE = "data/training_stats.json"

os.makedirs("logs", exist_ok=True)


# =====================================================
# LOAD ML MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("models/student_model.pkl")


try:
    model = load_model()
    model_status = True

except Exception as e:
    model = None
    model_status = False
    model_error = str(e)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "prediction": None,
    "risk": None,
    "student_data": None,
    "recommendations": None,
    "nav_page": "Overview"
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def load_history():

    if os.path.exists(LOG_FILE):

        try:
            return pd.read_csv(LOG_FILE)

        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def get_risk_color(risk):

    colors = {
        "Low": "#4ADE80",
        "Moderate": "#FBBF24",
        "High": "#F87171"
    }

    return colors.get(risk, "#60A5FA")


def metric_card(title, value, subtitle):

    render_html(f"""
        <div class="kpi-card">
            <div class="kpi-title">{html.escape(str(title))}</div>
            <div class="kpi-value">{html.escape(str(value))}</div>
            <div class="kpi-sub">{html.escape(str(subtitle))}</div>
        </div>
    """)


def save_prediction(student_data, prediction, risk):

    log_data = {
        **student_data,
        "predicted_score": prediction,
        "risk": risk,
        "timestamp": datetime.now().isoformat()
    }

    pd.DataFrame([log_data]).to_csv(
        LOG_FILE,
        mode="a",
        header=not os.path.exists(LOG_FILE),
        index=False
    )


def go_to_page(page):

    st.session_state.nav_page = page
    st.rerun()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    render_html("""
        <div style="margin-bottom:25px;">

            <div style="
                font-size:25px;
                font-weight:800;
                color:white;
            ">
                🎓 EduPredict
                <span style="color:#818CF8">AI</span>
            </div>

            <div style="
                color:#94A3B8;
                font-size:14px;
                margin-top:8px;
            ">
                Student Intelligence Platform
            </div>

        </div>
    """)

    st.divider()

    st.markdown("### NAVIGATION")

    pages = [
        "Overview",
        "Student Prediction",
        "AI Insights",
        "MLOps Monitoring",
        "Prediction History"
    ]

    current_page = st.session_state.nav_page

    if current_page not in pages:
        current_page = "Overview"

    page = st.radio(
        "Navigation",
        pages,
        index=pages.index(current_page),
        label_visibility="collapsed",
        key="navigation_radio"
    )

    st.session_state.nav_page = page

    st.divider()

    st.markdown("### SYSTEM STATUS")

    if model_status:
        st.success("● ML Model Active")
    else:
        st.error("● ML Model Error")

    st.caption("Version 3.0.0")
    st.caption("Random Forest Algorithm")
    st.caption("Powered by Machine Learning")

    st.divider()

    st.caption("© 2026 EduPredict AI")


# =====================================================
# GLOBAL DATA
# =====================================================

history = load_history()

total_predictions = len(history)

if not history.empty:

    average_score = history["predicted_score"].mean()

    high_risk = len(
        history[history["risk"] == "High"]
    )

else:

    average_score = 0
    high_risk = 0


# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "Overview":

    render_html("""
        <div class="hero">

            <div class="hero-badge">
                ● AI SYSTEM ONLINE
            </div>

            <div class="hero-title">
                Welcome to EduPredict AI
            </div>

            <div class="hero-description">
                Intelligent student performance prediction
                powered by Machine Learning and Generative AI.
                <br>
                Transform academic data into actionable insights.
            </div>

        </div>
    """)

    st.markdown("## Platform Overview")

    st.write("Your academic intelligence at a glance.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL PREDICTIONS",
            total_predictions,
            "Predictions generated"
        )

    with c2:
        metric_card(
            "AVERAGE SCORE",
            f"{average_score:.1f}",
            "Out of 100"
        )

    with c3:
        metric_card(
            "HIGH RISK STUDENTS",
            high_risk,
            "Require attention"
        )

    with c4:
        metric_card(
            "MODEL STATUS",
            "ACTIVE" if model_status else "ERROR",
            "Random Forest Model"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("## Quick Access")

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("### 🎯 Student Prediction")

            st.write(
                "Predict academic performance using "
                "your trained Machine Learning model."
            )

            if st.button(
                "Open Prediction →",
                key="quick_prediction",
                use_container_width=True
            ):
                go_to_page("Student Prediction")

    with c2:

        with st.container(border=True):

            st.markdown("### 🤖 AI Study Assistant")

            st.write(
                "Generate personalized academic "
                "recommendations using Generative AI."
            )

            if st.button(
                "Explore AI Insights →",
                key="quick_ai",
                use_container_width=True
            ):
                go_to_page("AI Insights")

    if not history.empty:

        st.markdown("## Recent Predictions")

        st.dataframe(
            history.tail(5).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# PAGE 2: STUDENT PREDICTION
# =====================================================

elif page == "Student Prediction":

    st.markdown("## 🎯 Student Performance Prediction")

    st.write(
        "Enter student information to predict academic performance."
    )

    st.divider()

    st.markdown(
        '<div class="section-label">STUDENT PARAMETERS</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.slider(
            "Attendance Percentage",
            35, 100, 80
        )

        study_hours = st.slider(
            "Daily Study Hours",
            0.0, 8.0, 3.0,
            step=0.5
        )

        previous_score = st.slider(
            "Previous Exam Score",
            30, 100, 65
        )

    with col2:

        assignment_completion = st.slider(
            "Assignment Completion",
            20, 100, 75
        )

        sleep_hours = st.slider(
            "Daily Sleep Hours",
            3.0, 10.0, 7.0,
            step=0.5
        )

    student_data = {

        "attendance": attendance,
        "study_hours": study_hours,
        "previous_score": previous_score,
        "assignment_completion": assignment_completion,
        "sleep_hours": sleep_hours

    }

    st.divider()

    st.markdown("### Student Profile")

    a, b, c = st.columns(3)

    with a:
        metric_card(
            "ATTENDANCE",
            f"{attendance}%",
            "Academic participation"
        )

    with b:
        metric_card(
            "STUDY HOURS",
            f"{study_hours} hrs",
            "Daily learning time"
        )

    with c:
        metric_card(
            "PREVIOUS SCORE",
            f"{previous_score}/100",
            "Previous performance"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🚀 Predict Student Performance",
        use_container_width=True
    ):

        if not model_status:

            st.error("ML model could not be loaded.")

        else:

            try:

                with st.spinner(
                    "Analyzing student performance..."
                ):

                    input_df = pd.DataFrame([student_data])

                    prediction = float(
                        model.predict(input_df)[0]
                    )

                    prediction = max(
                        0,
                        min(100, prediction)
                    )

                    if prediction >= 70:
                        risk = "Low"

                    elif prediction >= 50:
                        risk = "Moderate"

                    else:
                        risk = "High"

                    st.session_state.prediction = prediction

                    st.session_state.risk = risk

                    st.session_state.student_data = student_data

                    st.session_state.recommendations = None

                    save_prediction(
                        student_data,
                        prediction,
                        risk
                    )

                st.success(
                    "Prediction generated successfully!"
                )

            except Exception as e:

                st.error(f"Prediction failed: {e}")

    # DISPLAY RESULTS

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction

        risk = st.session_state.risk

        st.divider()

        st.markdown("## Prediction Results")

        c1, c2 = st.columns(2)

        with c1:

            render_html(f"""
                <div class="result-card">

                    <div class="result-label">
                        PREDICTED FINAL SCORE
                    </div>

                    <div class="result-score">
                        {prediction:.2f}
                    </div>

                    <div class="result-label">
                        OUT OF 100
                    </div>

                </div>
            """)

        with c2:

            color = get_risk_color(risk)

            render_html(f"""
                <div class="result-card">

                    <div class="result-label">
                        ACADEMIC RISK LEVEL
                    </div>

                    <div style="
                        font-size:40px;
                        font-weight:800;
                        color:{color};
                        margin:20px;
                    ">
                        {html.escape(risk)}
                    </div>

                    <div class="result-label">
                        Based on predicted performance
                    </div>

                </div>
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        st.progress(int(prediction))

        st.caption(
            f"Predicted academic performance: {prediction:.2f}%"
        )

        st.info(
            "Visit AI Insights to generate personalized study recommendations."
        )

        if st.button("Go to AI Insights →"):
            go_to_page("AI Insights")


# =====================================================
# PAGE 3: AI INSIGHTS
# =====================================================

elif page == "AI Insights":

    st.markdown("## 🤖 AI Study Assistant")

    st.write(
        "Get personalized academic recommendations "
        "powered by Generative AI."
    )

    st.divider()

    if st.session_state.prediction is None:

        st.warning(
            "Please generate a student prediction first."
        )

        if st.button("Go to Student Prediction →"):
            go_to_page("Student Prediction")

    else:

        render_html("""
            <div class="ai-card">

                <h3>✨ Personalized AI Analysis</h3>

                <p>
                    Gemini AI analyzes student performance
                    and generates personalized improvement strategies.
                </p>

            </div>
        """)

        prediction = st.session_state.prediction

        risk = st.session_state.risk

        c1, c2 = st.columns(2)

        c1.metric(
            "Predicted Score",
            f"{prediction:.2f}/100"
        )

        c2.metric(
            "Academic Risk",
            risk
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "✨ Generate AI Recommendations",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Gemini AI is analyzing student performance..."
                ):

                    recommendations = generate_recommendations(
                        st.session_state.student_data,
                        prediction
                    )

                    st.session_state.recommendations = recommendations

            except Exception as e:

                st.error(
                    f"AI service temporarily unavailable: {e}"
                )

        if st.session_state.recommendations:

            st.divider()

            st.markdown("### Your Personalized Study Plan")

            st.markdown(
                st.session_state.recommendations
            )


# =====================================================
# PAGE 4: MLOPS MONITORING
# =====================================================

elif page == "MLOps Monitoring":

    import json
    from pathlib import Path
    from mlops_monitor import run_drift_check

    st.markdown("## ⚙️ MLOps Monitoring Dashboard")

    st.write(
        "Monitor model performance, data drift "
        "and automated retraining."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("ML Model", "Random Forest")

    c2.metric("Experiment Tracking", "MLflow")

    c3.metric("Monitoring", "Evidently AI")

    st.divider()

    st.markdown("### Data Drift Detection")

    if st.button("Run Data Drift Check"):

        with st.spinner("Analyzing prediction data..."):

            try:

                st.session_state.drift_result = run_drift_check()

            except Exception as e:

                st.error(str(e))

    result = st.session_state.get("drift_result")

    if result:

        st.write("Status:", result["status"].upper())

        st.write(result["message"])

        st.metric(
            "Records Analyzed",
            result["rows_checked"]
        )

        if result["drift_detected"]:

            st.error("Data Drift Detected!")

            st.info(
                "Run retrain_pipeline.py to retrain the model."
            )

        elif result["status"] == "stable":

            st.success("Data Distribution Stable")

        else:

            st.warning("More prediction records required.")

    report = Path("reports/data_drift_report.html")

    if report.exists():

        with open(report, "rb") as f:

            st.download_button(
                "Download Drift Report",
                f,
                file_name="data_drift_report.html"
            )

    st.divider()

    st.markdown("### MLflow Experiment Tracking")

    st.write(
        "Track model training experiments, parameters "
        "and evaluation metrics."
    )

    st.code(
        "mlflow ui --backend-store-uri ./mlruns",
        language="bash"
    )

    st.info(
        "Open http://127.0.0.1:5000 "
        "after starting MLflow."
    )

    st.markdown("## ⚙️ MLOps Monitoring Dashboard")

    st.write(
        "Monitor model status and input data distribution."
    )

    st.divider()

    st.markdown("### Model Information")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model",
        "Random Forest"
    )

    c2.metric(
        "Algorithm",
        "Ensemble Learning"
    )

    c3.metric(
        "Status",
        "Active" if model_status else "Error"
    )

    st.divider()

    st.markdown("### Data Drift Monitoring")

    if os.path.exists(STATS_FILE):

        try:

            with open(STATS_FILE, "r") as f:
                stats = json.load(f)

            if st.session_state.student_data:

                unusual_features = []

                for feature, value in st.session_state.student_data.items():

                    if feature not in stats:
                        continue

                    mean = stats[feature]["mean"]

                    std = max(
                        stats[feature]["std"],
                        0.0001
                    )

                    if abs(value - mean) > 2 * std:

                        unusual_features.append(feature)

                if unusual_features:

                    st.warning(
                        "Unusual input detected: "
                        + ", ".join(unusual_features)
                    )

                else:

                    st.success(
                        "All input values are within normal training ranges."
                    )

            else:

                st.info(
                    "Generate a prediction to perform input distribution monitoring."
                )

        except Exception as e:

            st.error(f"Unable to read training statistics: {e}")

    else:

        st.warning(
            "Training statistics file not found."
        )

    st.divider()

    st.markdown("### MLflow Experiment Tracking")

    st.info(
        "MLflow experiment tracking is configured separately "
        "for monitoring model training runs."
    )

    render_html("""
        <div class="status-card">

            <b>● MODEL STATUS: OPERATIONAL</b>

            <br><br>

            ● Model training artifacts are available.

            <br><br>

            ● Random Forest model loaded.

            <br><br>

            ● Input distribution monitoring available.

        </div>
    """)


# =====================================================
# PAGE 5: PREDICTION HISTORY
# =====================================================

elif page == "Prediction History":

    st.markdown("## 📊 Prediction History")

    st.write(
        "Review previously generated student predictions."
    )

    st.divider()

    history = load_history()

    if not history.empty:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Predictions",
            len(history)
        )

        c2.metric(
            "Average Score",
            f"{history['predicted_score'].mean():.2f}"
        )

        c3.metric(
            "Highest Score",
            f"{history['predicted_score'].max():.2f}"
        )

        st.divider()

        st.markdown("### Performance Analytics")

        st.line_chart(
            history["predicted_score"],
            use_container_width=True
        )

        st.markdown("### Academic Risk Distribution")

        risk_count = history["risk"].value_counts()

        st.bar_chart(
            risk_count,
            use_container_width=True
        )

        st.divider()

        st.markdown("### All Prediction Records")

        st.dataframe(
            history.sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No prediction history available. "
            "Generate your first prediction!"
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

render_html("""
    <div class="footer">

        <b style="color:#A5B4FC;">
            EduPredict AI
        </b>

        | Intelligent Academic Analytics

        <br><br>

        Powered by Machine Learning, Generative AI & MLOps

        <br><br>

        © 2026 EduPredict AI. All Rights Reserved.

    </div>
""")