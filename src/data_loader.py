"""Load and clean the raw loan dataset."""
import pandas as pd
from src.config import RAW_DATA, PROCESSED_DATA, TARGET


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["Loan_ID"], errors="ignore")
    df = df.drop_duplicates()
    # Dependents may be '3+' in the public dataset
    if not pd.api.types.is_numeric_dtype(df["Dependents"]):
        df["Dependents"] = df["Dependents"].astype(str).str.replace("+", "", regex=False)
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")
    # Target -> 0/1
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Y": 1, "N": 0})
    return df


def load_clean(save=True) -> pd.DataFrame:
    df = clean(load_raw())
    if save:
        PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA, index=False)
    return df
