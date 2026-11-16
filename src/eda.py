"""Exploratory data analysis plots -> reports/figures.  Run: python -m src.eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import FIG_DIR, TARGET
from src.data_loader import load_clean


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    print(df.describe(include="all").T)
    print("\nMissing values:\n", df.isna().sum())

    df[TARGET].value_counts().plot(kind="bar", title="Loan Status (1=Approved)")
    plt.savefig(FIG_DIR / "eda_target.png", dpi=150); plt.close()

    df.groupby("Credit_History")[TARGET].mean().plot(kind="bar", title="Approval rate by Credit History")
    plt.savefig(FIG_DIR / "eda_credit_history.png", dpi=150); plt.close()

    df.groupby("Property_Area")[TARGET].mean().plot(kind="bar", title="Approval rate by Property Area")
    plt.savefig(FIG_DIR / "eda_property_area.png", dpi=150); plt.close()

    df["ApplicantIncome"].plot(kind="hist", bins=40, title="Applicant Income")
    plt.savefig(FIG_DIR / "eda_income.png", dpi=150); plt.close()
    print("EDA figures saved.")


if __name__ == "__main__":
    main()
