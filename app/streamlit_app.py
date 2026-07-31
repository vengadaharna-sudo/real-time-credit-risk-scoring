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

MODEL_PATH = os.path.join(BASE_DIR, "models")

OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")

# =============================================================================
# SECTION 4 — LOAD MODEL
# =============================================================================

best_model = joblib.load(
    os.path.join(MODEL_PATH, "best_model.pkl")
)

preprocessor = joblib.load(
    os.path.join(MODEL_PATH, "preprocessor.pkl")
)

# =============================================================================
# SECTION 5 — LOAD PREDICTIONS
# =============================================================================

results = pd.read_csv(
    os.path.join(
        OUTPUT_PATH,
        "credit_risk_predictions.csv"
    )
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
