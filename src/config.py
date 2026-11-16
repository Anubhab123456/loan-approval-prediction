"""Central configuration: paths, column names, constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "loan_data.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "loan_clean.csv"
MODEL_PATH = ROOT / "models" / "loan_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"

TARGET = "Loan_Status"  # 1 = Approved, 0 = Rejected
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History", "Dependents"]
CATEGORICAL_COLS = ["Gender", "Married", "Education",
                    "Self_Employed", "Property_Area"]
ENGINEERED_COLS = ["TotalIncome", "EMI", "Balance_Income", "LoanAmount_log"]
