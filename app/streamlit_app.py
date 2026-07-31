import streamlit as st

st.set_page_config(
    page_title="Real-Time Credit Risk Scoring",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Real-Time Credit Risk Scoring Framework")

st.markdown("""
## Retail Banking Decision Support System

Welcome to the Credit Risk Scoring Dashboard.

This application demonstrates the deployment of an XGBoost machine learning model for predicting loan default risk.

### Features

- 📊 Interactive Dashboard
- 🤖 Credit Risk Prediction
- ⚡ Real-Time Scoring Simulation
- 📈 Analytics & Visualisation
""")

st.success("Application loaded successfully!")
