"""Create a synthetic dataset with the same schema as the public
'Loan Prediction' dataset (Kaggle / Analytics Vidhya).
If you have the real CSV, just save it as data/raw/loan_data.csv
(Loan_Status Y/N is handled automatically) and skip this script.
"""
import numpy as np
import pandas as pd
from src.config import RAW_DATA, RANDOM_STATE


def generate(n=2000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Gender": rng.choice(["Male", "Female"], n, p=[0.8, 0.2]),
        "Married": rng.choice(["Yes", "No"], n, p=[0.65, 0.35]),
        "Dependents": rng.choice([0, 1, 2, 3], n, p=[0.57, 0.17, 0.16, 0.10]),
        "Education": rng.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22]),
        "Self_Employed": rng.choice(["Yes", "No"], n, p=[0.14, 0.86]),
        "ApplicantIncome": rng.lognormal(8.6, 0.55, n).round(),
        "CoapplicantIncome": np.where(rng.random(n) < 0.4, 0,
                                      rng.lognormal(7.6, 0.7, n).round()),
        "Loan_Amount_Term": rng.choice([120, 180, 240, 300, 360, 480], n,
                                       p=[0.02, 0.05, 0.03, 0.05, 0.83, 0.02]),
        "Credit_History": rng.choice([1.0, 0.0], n, p=[0.84, 0.16]),
        "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n,
                                    p=[0.33, 0.38, 0.29]),
    })
    total = df.ApplicantIncome + df.CoapplicantIncome
    df["LoanAmount"] = (total * rng.uniform(0.02, 0.06, n)).round()  # in thousands

    emi = df.LoanAmount * 1000 / df.Loan_Amount_Term
    burden = emi / total
    score = (3.0 * df.Credit_History + 0.4 * (df.Property_Area == "Semiurban")
             + 0.2 * (df.Education == "Graduate") + 0.2 * (df.Married == "Yes")
             - 6.0 * burden - 0.1 * df.Dependents + rng.normal(0, 0.6, n) - 1.2)
    df["Loan_Status"] = np.where(1 / (1 + np.exp(-score)) > 0.5, "Y", "N")

    # inject realistic missing values
    for col, p in [("Gender", .02), ("Married", .01), ("Self_Employed", .05),
                   ("LoanAmount", .04), ("Credit_History", .08),
                   ("Loan_Amount_Term", .02)]:
        df.loc[rng.random(n) < p, col] = np.nan
    return df


if __name__ == "__main__":
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    d = generate()
    d.to_csv(RAW_DATA, index=False)
    print(f"Saved {len(d)} rows -> {RAW_DATA}")
    print(d.Loan_Status.value_counts(normalize=True))
