"""Feature engineering + sklearn preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import NUMERIC_COLS, CATEGORICAL_COLS, ENGINEERED_COLS


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds TotalIncome, EMI, Balance_Income, LoanAmount_log."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalIncome"] = X["ApplicantIncome"] + X["CoapplicantIncome"]
        # LoanAmount is in thousands
        X["EMI"] = X["LoanAmount"] * 1000 / X["Loan_Amount_Term"]
        X["Balance_Income"] = X["TotalIncome"] - X["EMI"]
        X["LoanAmount_log"] = np.log1p(X["LoanAmount"])
        return X


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", num, NUMERIC_COLS + ENGINEERED_COLS),
        ("cat", cat, CATEGORICAL_COLS),
    ])
