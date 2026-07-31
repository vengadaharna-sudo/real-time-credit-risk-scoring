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

    st.subheader("Loan Information")

    col1, col2 = st.columns(2)

    with col1:

        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=500,
            max_value=40000,
            value=10000
        )

        term = st.selectbox(
            "Loan Term",
            [" 36 months", " 60 months"]
        )

        int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=40.0,
            value=12.0
        )

        installment = st.number_input(
            "Monthly Installment",
            min_value=0.0,
            value=300.0
        )

        purpose = st.selectbox(
            "Loan Purpose",
            [
                "car",
                "credit_card",
                "debt_consolidation",
                "educational",
                "home_improvement",
                "house",
                "major_purchase",
                "medical",
                "moving",
                "other",
                "renewable_energy",
                "small_business",
                "vacation",
                "wedding"
            ]
        )

        application_type = st.selectbox(
            "Application Type",
            [
                "Individual",
                "Joint App"
            ]
        )

    with col2:

        annual_inc = st.number_input(
            "Annual Income",
            min_value=1000,
            value=50000
        )

        dti = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            value=15.0
        )

        delinq_2yrs = st.number_input(
            "Delinquencies (2 Years)",
            min_value=0,
            value=0
        )

        open_acc = st.number_input(
            "Open Credit Accounts",
            min_value=0,
            value=10
        )

        total_acc = st.number_input(
            "Total Credit Accounts",
            min_value=1,
            value=15
        )

        mort_acc = st.number_input(
            "Mortgage Accounts",
            min_value=0,
            value=1
        )

    st.subheader("Credit Profile")

    col3, col4 = st.columns(2)

    with col3:

        fico_low = st.number_input(
            "FICO Range Low",
            min_value=300,
            max_value=850,
            value=680
        )

        fico_high = st.number_input(
            "FICO Range High",
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

        pub_rec = st.number_input(
            "Public Records",
            min_value=0,
            value=0
        )

        pub_rec_bankruptcies = st.number_input(
            "Public Record Bankruptcies",
            min_value=0,
            value=0
        )

    with col4:

        grade = st.selectbox(
            "Loan Grade",
            [
                "A","B","C","D","E","F","G"
            ]
        )

        sub_grade = st.selectbox(
            "Sub Grade",
            [
                "A1","A2","A3","A4","A5",
                "B1","B2","B3","B4","B5",
                "C1","C2","C3","C4","C5",
                "D1","D2","D3","D4","D5",
                "E1","E2","E3","E4","E5",
                "F1","F2","F3","F4","F5",
                "G1","G2","G3","G4","G5"
            ]
        )

        emp_length = st.selectbox(
            "Employment Length",
            [
                "< 1 year",
                "1 year",
                "2 years",
                "3 years",
                "4 years",
                "5 years",
                "6 years",
                "7 years",
                "8 years",
                "9 years",
                "10+ years"
            ]
        )

        home_ownership = st.selectbox(
            "Home Ownership",
            [
                "ANY",
                "MORTGAGE",
                "NONE",
                "OTHER",
                "OWN",
                "RENT"
            ]
        )

        verification_status = st.selectbox(
            "Verification Status",
            [
                "Not Verified",
                "Source Verified",
                "Verified"
            ]
        )

    submitted = st.form_submit_button(
        "Predict Credit Risk"
    )

# =============================================================================
# SECTION 13 — MAKE PREDICTION
# =============================================================================

if submitted:

    # Create input dataframe

    input_data = pd.DataFrame({

        "loan_amnt": [loan_amnt],
        "int_rate": [int_rate],
        "installment": [installment],
        "annual_inc": [annual_inc],
        "dti": [dti],
        "delinq_2yrs": [delinq_2yrs],
        "fico_range_low": [fico_low],
        "fico_range_high": [fico_high],
        "open_acc": [open_acc],
        "pub_rec": [pub_rec],
        "revol_bal": [revol_bal],
        "revol_util": [revol_util],
        "total_acc": [total_acc],
        "mort_acc": [mort_acc],
        "pub_rec_bankruptcies": [pub_rec_bankruptcies],

        "term": [term],
        "grade": [grade],
        "sub_grade": [sub_grade],
        "emp_length": [emp_length],
        "home_ownership": [home_ownership],
        "verification_status": [verification_status],
        "purpose": [purpose],
        "application_type": [application_type]

    })

    # Transform

    transformed = preprocessor.transform(input_data)

    # Predict

    prediction = best_model.predict(transformed)[0]

    probability = best_model.predict_proba(transformed)[0][1]

    # Risk Level

    if probability < 0.20:

        risk = "Low"

        recommendation = "Approve"

    elif probability < 0.40:

        risk = "Moderate"

        recommendation = "Review"

    elif probability < 0.60:

        risk = "High"

        recommendation = "Review"

    else:

        risk = "Very High"

        recommendation = "Reject"

    st.success("Prediction Complete")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Probability of Default",
        f"{probability:.2%}"
    )

    col2.metric(
        "Risk Level",
        risk
    )

    col3.metric(
        "Recommendation",
        recommendation
    )
