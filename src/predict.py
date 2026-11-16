"""Load the saved model and predict for one applicant."""
import joblib
import pandas as pd
from src.config import MODEL_PATH

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(applicant: dict) -> dict:
    proba = float(get_model().predict_proba(pd.DataFrame([applicant]))[0, 1])
    return {"approved": proba >= 0.5, "probability": proba}


if __name__ == "__main__":
    sample = dict(Gender="Male", Married="Yes", Dependents=1, Education="Graduate",
                  Self_Employed="No", ApplicantIncome=5000, CoapplicantIncome=1500,
                  LoanAmount=120, Loan_Amount_Term=360, Credit_History=1.0,
                  Property_Area="Urban")
    print(predict_one(sample))
