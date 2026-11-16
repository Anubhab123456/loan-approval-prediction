import pandas as pd
from src.generate_data import generate
from src.data_loader import clean
from src.preprocessing import FeatureEngineer
from src.train import make_pipeline, get_models


def test_clean_target_binary():
    df = clean(generate(200))
    assert set(df["Loan_Status"].unique()) <= {0, 1}


def test_feature_engineering_columns():
    X = clean(generate(50)).drop(columns=["Loan_Status"])
    out = FeatureEngineer().transform(X)
    for c in ["TotalIncome", "EMI", "Balance_Income", "LoanAmount_log"]:
        assert c in out.columns


def test_pipeline_fit_predict():
    df = clean(generate(300))
    X, y = df.drop(columns=["Loan_Status"]), df["Loan_Status"]
    pipe = make_pipeline(get_models()["LogisticRegression"]).fit(X, y)
    assert len(pipe.predict(X)) == len(X)
