# =============================================================================
# REAL-TIME CREDIT RISK SCORING FRAMEWORK
# Master's Capstone Project
# =============================================================================

# =============================================================================
# SECTION 1 — IMPORT LIBRARIES
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px

# =============================================================================
# SECTION 2 — PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Real-Time Credit Risk Scoring",
    page_icon="🏦",
    layout="wide"
)

# =============================================================================
# SECTION 3 — PROJECT PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = BASE_DIR

OUTPUT_PATH = BASE_DIR

# =============================================================================
# SECTION 4 — LOAD MODEL
# =============================================================================

best_model = joblib.load(
    os.path.join(BASE_DIR, "best_model.pkl")
)

preprocessor = joblib.load(
    os.path.join(BASE_DIR, "preprocessor.pkl")
)

# =============================================================================
# SECTION 5 — LOAD PREDICTIONS
# =============================================================================

results = pd.read_csv(
    os.path.join(BASE_DIR, "credit_risk_predictions.csv")
)
# =============================================================================
# SECTION 6 — PAGE TITLE
# =============================================================================

st.title("🏦 Real-Time Credit Risk Scoring Framework")

st.markdown(
"""
### Retail Banking Decision Support System

This dashboard demonstrates the deployment of the machine learning
credit risk model developed using the Lending Club dataset.

**Machine Learning Model:** XGBoost

**Methodology:** CRISP-DM

"""
)

# =============================================================================
# SECTION 7 — CALCULATE KPI METRICS
# =============================================================================

total_applications = len(results)

average_pd = results["Default_Probability"].mean()

high_risk = (results["Risk_Level"] == "High").sum()

very_high_risk = (results["Risk_Level"] == "Very High").sum()

approval_rate = (
    (results["Recommendation"] == "Approve").sum()
    / total_applications
) * 100

# =============================================================================
# SECTION 8 — KPI DASHBOARD
# =============================================================================

st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Applications",
    f"{total_applications:,}"
)

col2.metric(
    "Average PD",
    f"{average_pd:.2%}"
)

col3.metric(
    "High Risk",
    high_risk
)

col4.metric(
    "Very High Risk",
    very_high_risk
)

col5.metric(
    "Approval Rate",
    f"{approval_rate:.1f}%"
)

st.markdown("---")

# =============================================================================
# SECTION 9 — RISK LEVEL DISTRIBUTION
# =============================================================================

st.subheader("📊 Risk Level Distribution")

risk_counts = (
    results["Risk_Level"]
    .value_counts()
    .reset_index()
)

risk_counts.columns = ["Risk Level", "Applications"]

fig_risk = px.bar(
    risk_counts,
    x="Risk Level",
    y="Applications",
    text="Applications",
    title="Distribution of Predicted Credit Risk Levels"
)

fig_risk.update_traces(textposition="outside")

fig_risk.update_layout(
    xaxis_title="Risk Level",
    yaxis_title="Number of Applications"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

# =============================================================================
# SECTION 10 — DEFAULT PROBABILITY DISTRIBUTION
# =============================================================================

st.subheader("📈 Default Probability Distribution")

fig_prob = px.histogram(
    results,
    x="Default_Probability",
    nbins=30,
    title="Distribution of Predicted Default Probabilities"
)

fig_prob.update_layout(
    xaxis_title="Probability of Default",
    yaxis_title="Number of Applications"
)

st.plotly_chart(
    fig_prob,
    use_container_width=True
)

# =============================================================================
# SECTION 11 — PREDICTION RESULTS
# =============================================================================

st.subheader("📋 Latest Prediction Results")

st.dataframe(
    results,
    use_container_width=True,
    height=500
)

# =============================================================================
# SECTION 12 — MANUAL CREDIT RISK PREDICTION
# =============================================================================

st.markdown("---")
st.header("📝 Manual Credit Risk Prediction")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=500,
            max_value=40000,
            value=10000
        )

        annual_inc = st.number_input(
            "Annual Income",
            min_value=1000,
            value=50000
        )

        int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=40.0,
            value=12.0
        )

        dti = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            value=15.0
        )

        installment = st.number_input(
            "Installment",
            min_value=0.0,
            value=300.0
        )

    with col2:

        fico_low = st.number_input(
            "FICO Low",
            min_value=300,
            max_value=850,
            value=680
        )

        fico_high = st.number_input(
            "FICO High",
            min_value=300,
            max_value=850,
            value=684
        )

        revol_bal = st.number_input(
            "Revolving Balance",
            min_value=0,
            value=5000
        )

        revol_util = st.number_input(
            "Revolving Utilisation (%)",
            min_value=0.0,
            max_value=150.0,
            value=35.0
        )

        total_acc = st.number_input(
            "Total Accounts",
            min_value=1,
            value=15
        )

    submitted = st.form_submit_button("Predict Credit Risk")

# =============================================================================
# SECTION 13 — MAKE PREDICTION
# =============================================================================

if submitted:

    st.info(
        "Prediction functionality will be connected to the trained XGBoost model in the next section."
    )

