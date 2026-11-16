"""Streamlit web UI.  Run:  streamlit run app/app.py"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.predict import predict_one

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Prediction System")
st.caption("Enter applicant details to estimate loan approval chances.")

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["No", "Yes"])
    area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
with c2:
    app_inc = st.number_input("Applicant Income (monthly)", 0, 200000, 5000, step=500)
    co_inc = st.number_input("Co-applicant Income (monthly)", 0, 100000, 0, step=500)
    loan_amt = st.number_input("Loan Amount (in thousands)", 1, 1000, 120)
    term = st.selectbox("Loan Term (months)", [120, 180, 240, 300, 360, 480], index=4)
    credit = st.selectbox("Credit History (1 = good)", [1.0, 0.0])

if st.button("Predict", type="primary"):
    try:
        res = predict_one(dict(
            Gender=gender, Married=married, Dependents=dependents, Education=education,
            Self_Employed=self_emp, ApplicantIncome=app_inc, CoapplicantIncome=co_inc,
            LoanAmount=loan_amt, Loan_Amount_Term=term, Credit_History=credit,
            Property_Area=area))
        if res["approved"]:
            st.success(f"✅ Likely APPROVED (confidence {res['probability']:.1%})")
        else:
            st.error(f"❌ Likely REJECTED (approval probability {res['probability']:.1%})")
    except FileNotFoundError:
        st.warning("Model not found. Run `python -m src.train` first.")
